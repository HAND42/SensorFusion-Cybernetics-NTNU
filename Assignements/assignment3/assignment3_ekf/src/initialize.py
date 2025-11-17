import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu


def get_init_CV_state(meas0: Measurement2d, meas1: Measurement2d,
                      ekf_params: EKFParams) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt
    z0, z1 = meas0.value, meas1.value
    sigma_a = ekf_params.sigma_a
    sigma_z = ekf_params.sigma_z

    print(sigma_a, sigma_z)

    K = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1/dt, 0, -1/dt, 0],
        [0, 1/dt, 0, -1/dt]
    ])

    z = np.concatenate((z1.T, z0.T))
    mean = K @ z

    # print("mean_test", mean)

    Q = np.array([
        [dt**3 / 3, 0, dt**2 / 2, 0],
        [0, dt**3 / 3, 0, dt**2 / 2],
        [dt**2 / 2, 0, dt, 0],
        [0, dt**2 / 2, 0, dt]
    ])
    R = pow(sigma_z, 2) * np.eye(2)
    # cov = Q * pow(sigma_a, 2) + R

    #

    cov = np.array([[R, (1/dt)*R],
                    [(1/dt)*R, (2/(dt**2))*R + dt/3*np.eye(2)*sigma_a**2]])

    print(np.round(cov, 4))

    # init_state = MultiVarGauss(mean, cov)
    # print(init_state.cov)
    # print(init_state.mean)

    # TODO replace this with own code
    init_state = initialize_solu.get_init_CV_state(meas0, meas1, ekf_params)
    print(init_state.cov)
    # print(init_state.mean)
    return init_state
