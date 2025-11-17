import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:

    pred_meas = sens_modl.H @ state.mean
    S = sens_modl.H @ state.cov @ sens_modl.H.T + sens_modl.R
    kalman_gain = state.cov @ sens_modl.H.T @ np.linalg.inv(S)
    innovation = meas.value - pred_meas

    cond_mean = state.mean + \
        state.cov @ np.linalg.inv(state.cov +
                                  sens_modl.R) @ (meas.value - state.mean)
    cond_cov = state.cov - \
        state.cov @ np.linalg.inv(state.cov + sens_modl.R) @ state.cov

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)

    # print("ourrrrcond_state", cond_state)

    # cond_state = conditioning_solu.get_cond_state(state, sens_modl, meas)

    # print("truthhhcond_state", cond_state)

    return cond_state
