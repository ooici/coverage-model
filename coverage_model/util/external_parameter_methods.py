__author__ = 'casey'

import numpy as np


def linear_map(pval_callback, cov, external_param_name, time_segment):
    # TODO: Might not want to hard-code time
    x = pval_callback('time', time_segment).get_data()['time']
    x_i = cov.get_parameter_values('time', time_segment=time_segment).get_data()['time']
    y_i = cov.get_parameter_values(external_param_name, time_segment=time_segment).get_data()[external_param_name]


    # Where in x_i does x fit in?
    upper = np.searchsorted(x_i, x)
    # Clip values not in [1, N-1]
    upper = upper.clip(1, len(x_i)-1).astype(int)
    lower = upper - 1

    # Linear interpolation
    w = (x - x_i[lower]) / (x_i[upper] - x_i[lower])
    y = y_i[lower] * (1-w) + y_i[upper] * w
    return y
