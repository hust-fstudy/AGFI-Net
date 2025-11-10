import numpy as np


class CalGraph:
    def __init__(self, args, t_scale):
        self.args = args
        self.t_scale = t_scale
        self.grid_size = args.grid_size

    def __call__(self, events_dict: dict):
        data_dict = dict()
        data_dict['feat'] = np.hstack((events_dict['t'].reshape(-1, 1) / self.t_scale,
                                       events_dict['p'].reshape(-1, 1))).astype(np.float32)  # [n_t, p]
        data_dict['coord'] = np.hstack((events_dict['x'].reshape(-1, 1),
                                        events_dict['y'].reshape(-1, 1),
                                        events_dict['t'].reshape(-1, 1))).astype(np.float32)
        data_dict['label'] = events_dict['label']
        return data_dict
