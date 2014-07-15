__author__ = 'casey'

import numpy as np


def linear_map(pval_callback, cov, external_param_name, time_segment):
    # TODO: Might not want to hard-code time
    x = pval_callback('time', time_segment).get_data()['time']
    data_dict = cov.get_parameter_values(['time', external_param_name], time_segment=time_segment, fill_empty_params=True).get_data()
    if 'time' not in data_dict:
        return np.ones_like(x) * np.nan
    x_i = data_dict['time']
    y_i = data_dict[external_param_name]


    # Where in x_i does x fit in?
    upper = np.searchsorted(x_i, x)
    # Clip values not in [1, N-1]
    upper = upper.clip(1, len(x_i)-1).astype(int)
    lower = upper - 1

    # Linear interpolation
    w = (x - x_i[lower]) / (x_i[upper] - x_i[lower])
    y = y_i[lower] * (1-w) + y_i[upper] * w
    return y
