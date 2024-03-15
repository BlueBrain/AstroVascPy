"""Module dedicated to the report reading.
This is the standalone version before having proper sonata simulation config etc. The final version
will use snap and will be in archngv most probably.
Also this is a bit over-engineered but makes things easier if we have different types of reports
in the future. Plus, this is a light version of the snap classes which will make things easy to
adapt from these classes to the snap ones.
Copyright (c) 2023-2024 Blue Brain Project/EPFL
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd
from cached_property import cached_property
from libsonata import ElementReportReader, SonataError

from .exceptions import BloodFlowError
from .utils import ensure_list

# pylint: disable=missing-kwoa


def _collect_population_reports(frame_report, cls):
    return {
        population: cls(frame_report, population) for population in frame_report.population_names
    }


class PopulationFrameReport:
    """Access to PopulationFrameReport data.

    This function is generic for the soma-like or compartment-like reports.
    """

    def __init__(self, frame_report, population_name):
        """Initialize a PopulationFrameReport object from a FrameReport.

        Args:
            frame_report (FrameReport): FrameReport containing this frame report population.
            population_name (str): the population name corresponding to this report.

        Returns:
            PopulationFrameReport: A PopulationFrameReport object.
        """
        self.frame_report = frame_report
        self._frame_population = ElementReportReader(frame_report.filepath)[population_name]
        self._population_name = population_name

    @property
    def name(self):
        """Access to the population name."""
        return self._population_name

    @staticmethod
    def _wrap_columns(columns):
        """Allow to change the columns names if needed."""
        return columns

    def get(self, group=None, t_start=None, t_stop=None):
        """Fetch data from the report.

        Args:
            group (int/list/np.array): Get frames filtered by ids.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            pandas.DataFrame: frame as columns indexed by timestamps.
        """
        group = group if group is None else ensure_list(group)
        try:
            view = self._frame_population.get(node_ids=group, tstart=t_start, tstop=t_stop)
        except (SonataError, TypeError) as e:
            raise BloodFlowError(e) from e

        if len(view.ids) == 0:
            return pd.DataFrame()

        res = pd.DataFrame(
            data=view.data,
            columns=pd.MultiIndex.from_arrays(np.asarray(view.ids).T),
            index=view.times,
        ).sort_index(axis=1)

        # rename from multi index to index cannot be achieved easily through df.rename
        res.columns = self._wrap_columns(res.columns)
        return res

    @cached_property
    def node_ids(self):
        """Return the node ids present in the report.

        Returns:
            np.Array: Numpy array containing the node_ids included in the report
        """
        return np.sort(self._frame_population.get_node_ids())


class FilteredFrameReport:
    """Access to filtered FrameReport data."""

    def __init__(self, frame_report, group=None, t_start=None, t_stop=None):
        """Initialize a FilteredFrameReport.

        A FilteredFrameReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            frame_report (FrameReport): The FrameReport to filter.
            group (None/int/list/np.array/dict): Get frames filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredFrameReport: A FilteredFrameReport object.
        """
        self.frame_report = frame_report
        self.group = group
        self.t_start = t_start
        self.t_stop = t_stop

    @cached_property
    def report(self):
        """Access to the report data.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the report. Row's indices are the
                different timestamps and the column's MultiIndex are :
                - (population_name, node_id, compartment id) for the CompartmentReport
                - (population_name, node_id) for the SomaReport
        """
        res = pd.DataFrame()
        for population in self.frame_report.population_names:
            frames = self.frame_report[population]
            data = frames.get(group=self.group, t_start=self.t_start, t_stop=self.t_stop)
            if data.empty:
                continue
            new_index = tuple(tuple([population] + ensure_list(x)) for x in data.columns)
            data.columns = pd.MultiIndex.from_tuples(new_index)
            # need to do this in order to preserve MultiIndex for columns
            res = data if res.empty else data.join(res, how="outer")
        return res.sort_index().sort_index(axis=1)


class FrameReport:
    """Access to FrameReport data."""

    def __init__(self, filepath):
        """Initialize a FrameReport object from a filepath.

        Args:
            filepath (str/Path): path to the file containing the report

        Returns:
            FrameReport: A FrameReport object.
        """
        self.filepath = filepath

    @cached_property
    def _frame_reader(self):
        """Access to the compartment report reader."""
        return ElementReportReader(self.filepath)

    @property
    def time_units(self):
        """Return the data unit for this report."""
        units = {self._frame_reader[pop].time_units for pop in self.population_names}
        if len(units) > 1:
            raise BloodFlowError("Multiple time units found in the different populations.")
        return units.pop()

    @cached_property
    def data_units(self):
        """Return the data unit for this report."""
        units = {self._frame_reader[pop].data_units for pop in self.population_names}
        if len(units) > 1:
            raise BloodFlowError("Multiple data units found in the different populations.")
        return units.pop()

    @cached_property
    def population_names(self):
        """Return the population names included in this report."""
        return sorted(self._frame_reader.get_population_names())

    @cached_property
    def _population_report(self):
        """Collect the different PopulationFrameReport."""
        return _collect_population_reports(self, PopulationFrameReport)

    def __getitem__(self, population_name):
        """Access the PopulationFrameReports corresponding to the population 'population_name'."""
        return self._population_report[population_name]

    def __iter__(self):
        """Allow iteration over the different PopulationFrameReports."""
        return iter(self._population_report)

    def filter(self, group=None, t_start=None, t_stop=None):
        """Return a FilteredFrameReport.

        A FilteredFrameReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            group (None/int/list/np.array/dict): Get frames filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredFrameReport: A FilteredFrameReport object.
        """
        return FilteredFrameReport(self, group, t_start, t_stop)


class PopulationBloodflowReport(PopulationFrameReport):
    """Access to PopulationBloodflowReport data."""

    @staticmethod
    def _wrap_columns(columns):
        """Transform pandas.MultiIndex into pandas.Index for the pandas.DataFrame columns.

        Notes:
            the libsonata.ElementsReader.get() returns tuple as columns for the data. For the
            soma reports it means: pandas.MultiIndex([(0, 0), (1, 0), ..., (last_node_id, 0)]).
            So we convert this into pandas.Index([0,1,..., last_node_id]).
        """
        return columns.levels[0]


class BloodflowReport(FrameReport):
    """Access to a BloodflowReport data."""

    @cached_property
    def _population_report(self):
        """Collect the different PopulationBloodflowReport."""
        return _collect_population_reports(self, PopulationBloodflowReport)
