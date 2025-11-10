# -*- coding: utf-8 -*-
# @Time: 2025/6/27
# @File: encode.py
# @Author: fwb
import torch
import spconv.pytorch as spconv
from addict import Dict
from serialization import encode, offset2batch, batch2offset
