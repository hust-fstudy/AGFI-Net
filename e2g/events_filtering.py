import numpy as np
import open3d as o3d


def trans(nb_neighbors, std_ratio, rd: o3d.geometry.PointCloud):
    assert isinstance(rd, o3d.geometry.PointCloud) is True
    cl, ind = rd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inner_cloud = rd.select_by_index(ind)
    filtering_data = np.asarray(inner_cloud.points)
    filtering_feat = np.asarray(inner_cloud.colors)
    filtering_events = np.concatenate([filtering_data, filtering_feat[:, 1].reshape((-1, 1))], axis=1)
    return filtering_events


def filtering(every_k_points, nb_neighbors, std_ratio, events_dict, method='uniform_filtering'):
    assert method in ['uniform_filtering', 'voxel_filtering'], f"The {method} not in filtering methods!"
    events_arr = np.hstack((events_dict['t'].reshape(-1, 1),
                            events_dict['x'].reshape(-1, 1),
                            events_dict['y'].reshape(-1, 1),
                            events_dict['p'].reshape(-1, 1))).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(events_arr[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(events_arr[:, [0, 3, 3]])
    if method == 'uniform_filtering':
        rd = pcd.uniform_down_sample(every_k_points=every_k_points)
        filtering_events = trans(nb_neighbors, std_ratio, rd)
    elif method == 'voxel_filtering':
        rd = pcd.voxel_down_sample(voxel_size=1)
        filtering_events = trans(nb_neighbors, std_ratio, rd)
    else:
        filtering_events = None
    assert filtering_events is not None, f"The {method} not in filtering methods!"
    events_dict.update({'t': filtering_events[:, 0],
                        'x': filtering_events[:, 1],
                        'y': filtering_events[:, 2],
                        'p': filtering_events[:, 3]})
    return events_dict


def adaptive(args, events_dict):
    if args.is_adaptive:
        num_events = events_dict['x'].shape[0]
        base_events = args.base_events
        if num_events > base_events:
            every_k_points = int(num_events // base_events)
            events_dict = filtering(every_k_points, args.nb_neighbors, args.std_ratio, events_dict)
    else:
        events_dict = filtering(args.every_k_points, args.nb_neighbors, args.std_ratio, events_dict)
    return events_dict
