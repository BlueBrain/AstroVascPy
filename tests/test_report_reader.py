from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import astrovascpy.report_reader as test_module
from astrovascpy.exceptions import BloodFlowError

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / "data/reporting"


class TestBloodflowReport:
    def setup_method(self):
        self.test_obj = test_module.BloodflowReport(TEST_DATA_DIR / "compartment_report.h5")
        self.test_wrong = test_module.BloodflowReport(
            TEST_DATA_DIR / "diff_unit_compartment_report.h5"
        )

    def test_time_units(self):
        assert self.test_obj.time_units == "ms"
        with pytest.raises(BloodFlowError):
            self.test_wrong.time_units

    def test_data_units(self):
        assert self.test_obj.data_units == "mV"
        with pytest.raises(BloodFlowError):
            self.test_wrong.data_units

    def test_population_names(self):
        assert sorted(self.test_obj.population_names) == ["default", "default2"]

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationBloodflowReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationBloodflowReport)

    def test_filter(self):
        filtered = self.test_obj.filter(group=None, t_start=0.3, t_stop=0.6)
        assert filtered.frame_report == self.test_obj
        assert filtered.t_start == 0.3
        assert filtered.t_stop == 0.6
        assert filtered.group is None
        assert isinstance(filtered, test_module.FilteredFrameReport)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.4, 0.5, 0.6]))
        assert filtered.report.columns.tolist() == [
            ("default", 0),
            ("default", 1),
            ("default", 2),
            ("default2", 0),
            ("default2", 1),
            ("default2", 2),
        ]

        filtered = self.test_obj.filter(group=[1], t_start=0.3, t_stop=0.6)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.4, 0.5, 0.6]))
        assert filtered.report.columns.tolist() == [("default", 1), ("default2", 1)]

        filtered = self.test_obj.filter(group=[0, 1, 2], t_start=0.3, t_stop=0.6)
        assert filtered.report.columns.tolist() == [
            ("default", 0),
            ("default", 1),
            ("default", 2),
            ("default2", 0),
            ("default2", 1),
            ("default2", 2),
        ]

        filtered = self.test_obj.filter(group=[], t_start=0.3, t_stop=0.6)
        assert filtered.report.empty


class TestPopulationBloodflowReport:
    def setup_method(self):
        self.test_obj = test_module.BloodflowReport(TEST_DATA_DIR / "compartment_report.h5")[
            "default"
        ]
        timestamps = np.linspace(0, 0.9, 10)
        data = {0: timestamps, 1: timestamps + 1, 2: timestamps + 2}
        self.df = pd.DataFrame(data=data, index=timestamps, columns=[0, 1, 2]).astype(np.float32)

    def test_name(self):
        assert self.test_obj.name == "default"

    def test_get(self):
        pdt.assert_frame_equal(self.test_obj.get(), self.df, check_column_type=False)
        pdt.assert_frame_equal(self.test_obj.get([]), pd.DataFrame())
        pdt.assert_frame_equal(self.test_obj.get(np.array([])), pd.DataFrame())
        pdt.assert_frame_equal(self.test_obj.get(()), pd.DataFrame())

        pdt.assert_frame_equal(self.test_obj.get(2), self.df.loc[:, [2]], check_column_type=False)

        pdt.assert_frame_equal(
            self.test_obj.get([2, 0]), self.df.loc[:, [0, 2]], check_column_type=False
        )

        pdt.assert_frame_equal(
            self.test_obj.get([0, 2]), self.df.loc[:, [0, 2]], check_column_type=False
        )

        pdt.assert_frame_equal(
            self.test_obj.get(np.asarray([0, 2])), self.df.loc[:, [0, 2]], check_column_type=False
        )

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_stop=0.5),
            self.df.iloc[:6].loc[:, [2]],
            check_column_type=False,
        )

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_stop=0.55),
            self.df.iloc[:6].loc[:, [2]],
            check_column_type=False,
        )

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_start=0.5),
            self.df.iloc[5:].loc[:, [2]],
            check_column_type=False,
        )

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_start=0.5, t_stop=0.8),
            self.df.iloc[5:9].loc[:, [2]],
            check_column_type=False,
        )

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8),
            self.df.iloc[5:9].loc[:, [1, 2]],
            check_column_type=False,
        )

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8),
            self.df.iloc[2:9].loc[:, [1, 2]],
            check_column_type=False,
        )

        with pytest.raises(BloodFlowError):
            self.test_obj.get(-1, t_start=0.2)

        with pytest.raises(BloodFlowError):
            self.test_obj.get(0, t_start=-1)

        with pytest.raises(BloodFlowError):
            self.test_obj.get([0, 2], t_start=15)

    def test_node_ids(self):
        npt.assert_array_equal(self.test_obj.node_ids, np.array(sorted([0, 1, 2])))


def test_overriden_function():
    test_obj = test_module.FrameReport(TEST_DATA_DIR / "compartment_report.h5")["default"]
    assert test_obj.get(group=0, t_start=None, t_stop=0.0).iloc[0][0][0] == 0.0
