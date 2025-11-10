# -*- coding: utf-8 -*-
# @Time: 2025/6/27
# @File: net.py
# @Author: fwb
import torch
import torch.nn as nn
import torch_scatter
from timm.layers import DropPath
from functools import partial
from layers.structure import Point
from layers.sequential import PointModule, PointSequential
from layers.embedding import Embedding
from layers.pooling import SerializedPooling
from layers.attention import SerializedAttention
from layers.classifier import Classifier

try:
    import flash_attn
except ImportError:
    flash_attn = None


class MLP(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
            self,
            channels,
            num_heads,
            patch_size=48,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            pre_norm=True,
            order_index=0,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=True,
            upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class Net(nn.Module):
    def __init__(self,
                 in_chs=2,
                 grid_size=(4, 3, 3),
                 num_classes=101,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 stride=(2, 2, 2),
                 enc_depths=(2, 2, 2, 2),
                 enc_chs=(32, 64, 128, 256),
                 enc_heads=(2, 4, 8, 16),
                 enc_patches=(64, 64, 64, 64),
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.3,
                 pre_norm=True,
                 shuffle_orders=True,
                 enable_rpe=False,
                 enable_flash=True,
                 upcast_attention=False,
                 upcast_softmax=False
                 ):
        super().__init__()
        self.grid_size = grid_size
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_chs)
        assert self.num_stages == len(enc_heads)
        assert self.num_stages == len(enc_patches)
        # Norm layers.
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        # Activation layers.
        act_layer = nn.GELU
        # Stem.
        self.embedding = Embedding(
            in_channels=in_chs,
            embed_channels=enc_chs[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )
        # Encoder.
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                             sum(enc_depths[:s]): sum(enc_depths[: s + 1])
                             ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_chs[s - 1],
                        out_channels=enc_chs[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_chs[s],
                        num_heads=enc_heads[s],
                        patch_size=enc_patches[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")
        self.cls = Classifier(enc_chs[-1], num_classes)

    def forward(self, data):
        data['feat'] = data.pop('x')
        data['coord'] = data.pop('pos')
        data['grid_size'] = torch.tensor(self.grid_size).to(data['feat'].device)
        data['feat'] = torch.cat([data['feat'], data['coord']], dim=1).to(data['feat'].device)
        point = Point(data)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        point.feat = torch_scatter.segment_csr(
            src=point.feat,
            indptr=nn.functional.pad(point.offset, (1, 0)),
            reduce="max",
        )
        out = self.cls(point.feat)
        return out
