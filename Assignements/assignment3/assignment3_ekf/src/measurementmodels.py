from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from mytypes import MultiVarGauss
from solution import measurementmodels as measurementmodels_solu


@dataclass
class CartesianPosition2D:
    sigma_z: float

    def h(self, x: ndarray) -> ndarray:
        """Calculate the noise free measurement value of x."""

        # TODO replace this with own code
        x_h = self.H(x) @ x.T

        # x_h = measurementmodels_solu.CartesianPosition2D.h(self, x)
        return x_h

    def H(self, x: ndarray) -> ndarray:
        """Get the measurement matrix at x."""

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # print("ourH", H)

        # TODO replace this with own code
        # H = measurementmodels_solu.CartesianPosition2D.H(self, x)

        # print("truthH", H)

        return H

    def R(self, x: ndarray) -> ndarray:
        """Get the measurement covariance matrix at x."""

        # TODO replace this with own code

        R = (self.sigma_z**2) * np.eye(2)

        # R = measurementmodels_solu.CartesianPosition2D.R(self, x)
        return R

    def predict_measurement(self,
                            state_est: MultiVarGauss
                            ) -> MultiVarGauss:
        """Get the predicted measurement distribution given a state estimate.
        See 4. and 6. of Algorithm 1 in the book.
        """
        z_hat = self.h(state_est.mean)
        H = self.H(state_est)
        S = H@state_est.cov@H.T + self.R(state_est.cov)

        measure_pred_gauss = MultiVarGauss(z_hat, S)

        # print("rtttt", measure_pred_gauss.cov)
        # print(measure_pred_gauss.mean)

        # TODO replace this with own code
        # measure_pred_gauss = measurementmodels_solu.CartesianPosition2D.predict_measurement(
        #     self, state_est)

        # print("truth", measure_pred_gauss.cov)
        # print(measure_pred_gauss.mean)

        return measure_pred_gauss
