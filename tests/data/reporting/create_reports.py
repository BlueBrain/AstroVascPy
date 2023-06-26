"""Taken from the libsonata lib."""
import h5py
import numpy as np


def write_element_report(filepath, units):
    population_names = ["default", "default2"]
    node_ids = np.arange(0, 3)
    index_pointers = np.arange(0, 4)
    element_ids = np.zeros(3)
    times = (0.0, 1.0, 0.1)
    data = [node_ids + j * 0.1 for j in range(10)]

    string_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(filepath, "w") as h5f:
        h5f.create_group("report")
        gpop_element = h5f.create_group("/report/" + population_names[0])
        ddata = gpop_element.create_dataset("data", data=data, dtype=np.float32)
        ddata.attrs.create("units", data=units[population_names[0]]["data"], dtype=string_dtype)
        gmapping = h5f.create_group("/report/" + population_names[0] + "/mapping")

        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset("time", data=times, dtype=np.double)
        dtimes.attrs.create("units", data=units[population_names[0]]["time"], dtype=string_dtype)

        gpop_element2 = h5f.create_group("/report/" + population_names[1])
        ddata = gpop_element2.create_dataset("data", data=data, dtype=np.float32)
        ddata.attrs.create("units", data=units[population_names[1]]["data"], dtype=string_dtype)
        gmapping = h5f.create_group("/report/" + population_names[1] + "/mapping")

        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset("time", data=times, dtype=np.double)
        dtimes.attrs.create("units", data=units[population_names[1]]["time"], dtype=string_dtype)


if __name__ == "__main__":
    units = {
        "default": {"time": "ms", "data": "mV"},
        "default2": {"time": "ms", "data": "mV"},
    }
    write_element_report("compartment_report.h5", units)
    units_diff = {
        "default": {"time": "ms", "data": "mV"},
        "default2": {"time": "s", "data": "mR"},
    }
    write_element_report("diff_unit_compartment_report.h5", units_diff)
