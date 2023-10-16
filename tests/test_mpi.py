import numpy as np
import numpy.testing as npt
import petsc4py
import pytest
from mpi4py import MPI
from scipy.sparse import coo_matrix

from astrovascpy.scipy_petsc_conversions import PETScMat2coo
from astrovascpy.scipy_petsc_conversions import PETScVec2array
from astrovascpy.scipy_petsc_conversions import array2PETScVec
from astrovascpy.scipy_petsc_conversions import coomatrix2PETScMat

# pip install pytest-mpi
# mpirun -n 4 pytest --with-mpi tests/test_mpi.py

# The tests are skipped if not using the option --with-mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@pytest.mark.mpi(min_size=2)
def test_numpy_array2petsc():
    """Test that the conversion from numpy array to petsc arrays works as expected"""

    # case 1
    vec1 = np.array([]) if rank == 0 else None
    temp1 = array2PETScVec(vec1)
    vec1_new = PETScVec2array(temp1)
    npt.assert_array_equal(vec1_new, vec1)

    # case 2
    vec2 = np.array([1.0]) if rank == 0 else None
    temp2 = array2PETScVec(vec2)
    vec2_new = PETScVec2array(temp2)
    npt.assert_array_equal(vec2_new, vec2)

    # case 3
    vec3 = np.arange(22, dtype=np.int64) if rank == 0 else None
    temp3 = array2PETScVec(vec3)
    vec3_new = PETScVec2array(temp3)
    # array2PETScVec converts automatically to float
    if rank == 0:
        assert vec3.dtype != vec3_new.dtype
        assert vec3.dtype == np.int64
        assert vec3_new.dtype == np.float64
        npt.assert_array_equal(vec3_new, vec3)


@pytest.mark.mpi(min_size=2)
def test_scipy2petsc_conversion():
    """Test that the conversion from scipy sparse coo to petsc arrays works as expected"""

    def create_sparse_coo(m, n, seed=33):
        """Create a coo matrix of size (m,n) with m elements in random positions"""
        np.random.seed(seed)
        X = 10  # just a random value
        # coo representation (row, col, data)
        row = np.arange(start=0, stop=m)
        col = np.random.randint(low=0, high=n, size=m)
        data = np.arange(start=X, stop=X + m)
        return coo_matrix((data, (row, col)), shape=(m, n))

    # case 1
    if rank == 0:
        A = create_sparse_coo(m=5, n=8, seed=31)
    else:
        A = None
    A_petsc = coomatrix2PETScMat(A)
    A_coo = PETScMat2coo(A_petsc)

    if rank == 0:
        npt.assert_array_equal(A.row, A_coo.row)
        npt.assert_array_equal(A.col, A_coo.col)
        npt.assert_array_equal(A.data, A_coo.data)

    # case 2
    if rank == 0:
        B = create_sparse_coo(m=1000, n=80000, seed=32)
    else:
        B = None
    B_petsc = coomatrix2PETScMat(B)
    B_coo = PETScMat2coo(B_petsc)

    if rank == 0:
        npt.assert_array_equal(B.row, B_coo.row)
        npt.assert_array_equal(B.col, B_coo.col)
        npt.assert_array_equal(B.data, B_coo.data)

    # case 3:  empty matrix
    if rank == 0:
        m = 10000
        n = 10000
        C = coo_matrix((m, n))
    else:
        C = None
    C_petsc = coomatrix2PETScMat(C)
    C_coo = PETScMat2coo(C_petsc)

    if rank == 0:
        npt.assert_array_equal(C.row, C_coo.row)
        npt.assert_array_equal(C.col, C_coo.col)
        npt.assert_array_equal(C.data, C_coo.data)


@pytest.mark.mpi(min_size=2)
def test_coomatrix2PETScMat():
    """The input of coomatrix2PETScMat must be a scipy sparse matrix
    and the output must be a PETSC object"""

    D = None
    if rank == 0:
        with pytest.raises(AttributeError):
            coomatrix2PETScMat(D)

    F = np.eye(N=10)
    if rank == 0:
        with pytest.raises(AttributeError):
            coomatrix2PETScMat(F)

    if rank == 0:
        m, n = 1000, 2000
        G = coo_matrix((m, n))
    else:
        G = None
    G_petsc = coomatrix2PETScMat(G)
    assert isinstance(G_petsc, petsc4py.PETSc.Mat)
