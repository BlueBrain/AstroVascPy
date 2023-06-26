from pathlib import Path

import numpy as np
import numpy.testing as npt

import astrovascpy.report_writer as test_module
from astrovascpy.report_reader import BloodflowReport

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / "data/reporting/export"

if not TEST_DATA_DIR.exists():
    TEST_DATA_DIR.mkdir()


def test_write_simulation_report():
    n_edges = 100
    start_time = 0
    end_time = 1.0
    time_step = 0.1
    nb_time_step = int((end_time - start_time) / time_step)
    flows = np.zeros((nb_time_step, n_edges))
    pressures = np.zeros((nb_time_step, n_edges))
    radii = np.ones((nb_time_step, n_edges))
    volumes = np.power(radii, 2) * np.pi * np.ones(n_edges)
    test_module.write_simulation_report(
        np.arange(n_edges),
        TEST_DATA_DIR,
        start_time,
        end_time,
        time_step,
        flows,
        pressures,
        radii,
        volumes,
    )
    report = BloodflowReport(TEST_DATA_DIR / "report_radii.h5")
    npt.assert_almost_equal(report["vasculature"].get(), radii)
    report = BloodflowReport(TEST_DATA_DIR / "report_flows.h5")
    npt.assert_almost_equal(report["vasculature"].get(), flows)
    report = BloodflowReport(TEST_DATA_DIR / "report_pressures.h5")
    npt.assert_almost_equal(report["vasculature"].get(), pressures)
    report = BloodflowReport(TEST_DATA_DIR / "report_volumes.h5")
    npt.assert_almost_equal(report["vasculature"].get(), volumes)
