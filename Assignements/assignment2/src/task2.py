import numpy as np
from scipy.stats import norm
from typing import Tuple

from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from conditioning import get_cond_state
from solution import task2 as task2_solu


def get_conds(state: MultiVarGauss2d,
              sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
              sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
              ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    cond_c = get_cond_state(state, sens_model_c, meas_c)
    cond_r = get_cond_state(state, sens_model_r, meas_r)

    # print("ourcondc", cond_c, "ourcondr", cond_r)

    # # TODO replace this with own code
    # cond_c, cond_r = task2_solu.get_conds(
    #     state, sens_model_c, meas_c, sens_model_r, meas_r)

    # print("truth condc", cond_c, "condr", cond_r)

    return cond_c, cond_r


def get_double_conds(state: MultiVarGauss2d,
                     sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
                     sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
                     ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    cond_rc = get_cond_state(get_cond_state(
        state, sens_model_r, meas_r), sens_model_c, meas_c)
    cond_cr = get_cond_state(get_cond_state(
        state, sens_model_c, meas_c), sens_model_r, meas_r)

    # print("ourrrr cond_cr, cond_rc", cond_cr, cond_rc)

    # # TODO replace this with own code
    # cond_cr, cond_rc = task2_solu.get_double_conds(
    #     state, sens_model_c, meas_c, sens_model_r, meas_r)

    # print("truth cond_cr, cond_rc", cond_cr, cond_rc)

    return cond_cr, cond_rc


def get_prob_over_line(gauss: MultiVarGauss2d) -> float:

    lin_transform = np.array([[-1, 1]])

    transfo = gauss.get_transformed(lin_transform)

    std = np.sqrt(np.diag(np.diag(transfo.cov)))

    prob = 1 - norm.cdf(5, transfo.mean, std)

    # print("our prob", prob)

    # # TODO replace this with own code
    # prob = task2_solu.get_prob_over_line(gauss)

    # print("thruth prob", prob)

    return prob
