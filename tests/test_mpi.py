import numpy as np
import numpy.testing as npt
import pytest

from astrovascpy.scipy_petsc_conversions import PETScVec2array
from astrovascpy.scipy_petsc_conversions import array2PETScVec

# pip install pytest-mpi
# mpirun -n 2 pytest --with-mpi tests/test_mpi.py


# This test is skipped if not using the option --with-mpi 
@pytest.mark.mpi(min_size=2)
def test_petsc2numpy_array():
    """Test that the conversion from numpy array to petsc arrays works as expected"""

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    vec1 = np.array([]) if rank == 0 else None
    temp1 = array2PETScVec(vec1)
    vec1_new = PETScVec2array(temp1)
    npt.assert_array_equal(vec1_new, vec1)

    vec2 = np.array([1.0]) if rank == 0 else None
    temp2 = array2PETScVec(vec2)
    vec2_new = PETScVec2array(temp2)
    npt.assert_array_equal(vec2_new, vec2)

    vec3 = np.arange(22) if rank == 0 else None
    temp3 = array2PETScVec(vec3)
    vec3_new = PETScVec2array(temp3)
    if rank == 0:
        assert vec3.dtype != vec3_new.dtype

    npt.assert_array_equal(vec3_new, vec3)
