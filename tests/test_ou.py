import numpy as np
import numpy.testing as npt

import astrovascpy.ou as OU


def test_OU_calibration():
    """Verify that the calibration is able to calculate the correct kappa and sigma"""

    def verify(kappa_test, sigma_test):
        r_max_test = 2.8 * sigma_test / np.sqrt(2 * kappa_test)
        target_time = OU.expected_time(kappa_test, r_max_test, C=2.8)
        kappa, sigma = OU.compute_OU_params(time=target_time, x_max=r_max_test, c=2.8)
        npt.assert_allclose((kappa, sigma), (kappa_test, sigma_test), rtol=1e-4, atol=1e-8)

    kappa_set = [0.1, 0.3, 1, 5, 10, 20]
    sigma_set = [0.2, 1.5, 4, 12]

    for k in kappa_set:
        for s in sigma_set:
            verify(k, s)
