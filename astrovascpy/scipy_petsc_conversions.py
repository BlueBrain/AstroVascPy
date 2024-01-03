"""
Copyright (c) 2023-2023 Blue Brain Project/EPFL
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
import os
import random
import string

from mpi4py import MPI
from numpy import concatenate
from numpy import dtype
from numpy import zeros as np_zeros
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from astrovascpy import PetscBinaryIO

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()


def _from_numpy_dtype(np_type):
    """Convert NumPy datatype to MPI datatype."""
    dtype_var = dtype(np_type)
    char_d = dtype_var.char
    mpi_type = MPI._typedict[char_d]
    return mpi_type


def BinaryIO2PETScMat(L, file_name="tempMat.dat"):
    """
    Args:
        L: Numpy/SciPy array/matrix [on process 0]
        file_name: file name for the temporary container

    Returns:
        PETScMat [distributed across ranks] ->
        it uses the PetscBinaryIO interface (read/write from/to disk)
    """

    # randomize the name of the temp container to avoid files with the same name,
    # when running more than two instances of the program
    fname_ = "".join(random.choices(string.ascii_lowercase, k=10))
    fname_ += f"_{str(id(L))}_" + file_name
    fname_ = MPI_COMM.bcast(fname_, root=0)

    if MPI_RANK == 0:
        PetscBinaryIO.PetscBinaryIO().writeBinaryFile(
            fname_,
            [
                L,
            ],
        )

    viewer = PETSc.Viewer().createBinary(fname_, "r")
    A = PETSc.Mat(comm=MPI_COMM).load(viewer)

    if MPI_RANK == 0:
        os.remove(fname_)
        try:
            os.remove(fname_ + ".info")
        except FileNotFoundError:
            pass

    return A


def BinaryIO2PETScVec(v, file_name="tempVec.dat"):
    """
    Args:
        v: Numpy array/vector [on process 0]
        file_name: file name for the temporary container

    Returns:
        PETSc Vec [distributed across ranks] ->
        it uses the PetscBinaryIO interface (read/write from/to disk)
    """

    fname_ = "".join(random.choices(string.ascii_lowercase, k=10)) + f"_{str(id(v))}_" + file_name
    fname_ = MPI_COMM.bcast(fname_, root=0)

    if MPI_RANK == 0:
        PetscBinaryIO.PetscBinaryIO().writeBinaryFile(
            fname_,
            [
                v.view(PetscBinaryIO.Vec),
            ],
        )

    viewer = PETSc.Viewer().createBinary(fname_, "r")
    x = PETSc.Vec(comm=MPI_COMM).load(viewer)

    if MPI_RANK == 0:
        os.remove(fname_)
        try:
            os.remove(fname_ + ".info")
        except FileNotFoundError:
            pass

    return x


def BinaryIO2array(x, file_name="tempVec.dat"):
    """
    Args:
        x: a distributed PETSc Vec
        file_name: file name for the temporary container

    Returns:
        numpy array on proc 0 -> it uses the PetscBinaryIO interface (read/write from/to disk)
    """

    fname_ = "".join(random.choices(string.ascii_lowercase, k=10)) + f"_{str(id(x))}_" + file_name
    fname_ = MPI_COMM.bcast(fname_, root=0)

    viewer = PETSc.Viewer().createBinary(fname_, "w")
    viewer(x)

    v = None
    if MPI_RANK == 0:
        (v,) = PetscBinaryIO.PetscBinaryIO().readBinaryFile(fname_)

    if MPI_RANK == 0:
        os.remove(fname_)
        try:
            os.remove(fname_ + ".info")
        except FileNotFoundError:
            pass

    return v


def coomatrix2PETScMat(L):
    """
    Converts a sequential scipy sparse matrix (on process 0) to a PETSc
    Mat ('aij') matrix distributed on all processes

    Args:
        L: scipy sparse matrix on proc 0 (COO format)

    Returns:
        PETSc matrix distributed on all procs
    """

    # Get the data from the sequential scipy matrix
    if MPI_RANK == 0:
        if L.format == "coo":
            L2 = L
        else:
            L2 = L.tocoo()

        n, m = L2.shape

        # COO-related
        row_ = L2.row
        row_ = row_.astype(PETSc.IntType)
        col_ = L2.col
        col_ = col_.astype(PETSc.IntType)
        data_ = L2.data
        data_ = data_.astype(PETSc.ScalarType)

        # CSR-related
        # https://en.wikipedia.org/wiki/Sparse_matrix
        # ROW_INDEX
        Ai = L.tocsr().indptr
        Ai = Ai.astype(PETSc.IntType)
    else:
        n = None
        m = None
        row_ = None
        col_ = None
        data_ = None
        Ai = None

    # Broadcast sizes
    n = MPI_COMM.bcast(n, root=0)
    m = MPI_COMM.bcast(m, root=0)

    A = PETSc.Mat().create(comm=MPI_COMM)
    A.setSizes([n, m])
    A.setType("aij")
    A.setFromOptions()
    A.setUp()

    # rows corresponding to the current mpi rank (range)
    istart, iend = A.getOwnershipRange()

    # gather all ranges in rank 0 (None for the other ranks)
    istart_loc = MPI_COMM.gather(istart, root=0)
    iend_loc = MPI_COMM.gather(iend, root=0)

    nnzloc = None
    if MPI_RANK == 0:
        nnzloc_0 = np_zeros(MPI_SIZE, PETSc.IntType)
        for i in range(MPI_SIZE):
            # Ai encodes the total number of nonzeros above row istart_loc[i] and iend_loc[i]
            # how many non-zero elements for rank i
            nnzloc_0[i] = Ai[iend_loc[i]] - Ai[istart_loc[i]]
    else:
        nnzloc_0 = None

    # every rank gets the corresponding number (from vector to number)
    nnzloc = MPI_COMM.scatter(nnzloc_0, root=0)

    # distribute the matrix across ranks (COO format) - create local containers
    row_loc = np_zeros(nnzloc, PETSc.IntType)
    col_loc = np_zeros(nnzloc, PETSc.IntType)
    data_loc = np_zeros(nnzloc, PETSc.ScalarType)

    # For Scatterv
    displ_ = None
    if MPI_RANK == 0:
        displ_ = tuple(concatenate(([0], nnzloc_0[:-1])).cumsum())

    # distribute the matrix across ranks (COO format) - populate local containers
    MPI_COMM.Scatterv([row_, nnzloc_0, displ_, _from_numpy_dtype(PETSc.IntType)], row_loc, root=0)
    MPI_COMM.Scatterv([col_, nnzloc_0, displ_, _from_numpy_dtype(PETSc.IntType)], col_loc, root=0)
    MPI_COMM.Scatterv(
        [data_, nnzloc_0, displ_, _from_numpy_dtype(PETSc.ScalarType)], data_loc, root=0
    )

    for r, c, v in zip(row_loc, col_loc, data_loc):
        A[r, c] = v

    A.assemble()

    return A


def _distribute_array_helper(v, array_type=None):
    """
    Scatter a NumPy array from rank 0 to all ranks using PETSc automatic
    chunk selection routine.

    Args:
        v: NumPy array on rank 0, None (or whatever) on other ranks
        array_type: set the type of the distributed array
                    If None, it keeps the same type as v.

    Returns:
        numpy.array: distributed array on all procs
        petsc4py.PETSc.Vec: distributed array on all procs. 
                            All entries are initialized to zero.
    """

    if MPI_RANK == 0:
        n = len(v)
        if array_type is None:
            array_type = v.dtype
        else:
            v = v.astype(array_type)
    else:
        n = None

    # Broadcast size and type
    n = MPI_COMM.bcast(n, root=0)
    array_type = MPI_COMM.bcast(array_type, root=0)

    # distribute array using PETSc.Vec approach
    x = PETSc.Vec()
    x.create(MPI_COMM)
    x.setSizes(n)
    x.setFromOptions()
    istart, iend = x.getOwnershipRange()

    # slice of the global vector that belongs to this mpi rank (range: from -> to)
    nloc = iend - istart
    # gather nloc on rank zero
    nloc_loc = MPI_COMM.gather(nloc, root=0)
    if MPI_RANK != 0:
        nloc_loc = [0]
    # gather istart on rank zero.
    istart_loc = MPI_COMM.gather(istart, root=0)
    if MPI_RANK != 0:
        istart_loc = [0]

    # Initialize destination array on each rank
    vloc = np_zeros(nloc, dtype=array_type)

    # scatter the vector v on all ranks
    MPI_COMM.Scatterv([v, nloc_loc, istart_loc, _from_numpy_dtype(array_type)], vloc, root=0)

    return vloc, x


def distribute_array(v, array_type=None):
    """
    Scatter a NumPy array from rank 0 to all ranks using PETSc automatic
    chunk selection routine.

    Args:
        v: NumPy array on rank 0, None (or whatever) on other ranks
        array_type: set the type of the distributed array
                    If None, it keeps the same type as v.

    Returns:
        numpy.array: distributed array on all procs
    """

    vloc, x = _distribute_array_helper(v, array_type=array_type)
    x.destroy()  # Free the memory of the PETSc vec

    return vloc


def array2PETScVec(v):
    """
    Converts (copies) a sequential array/vector on process 0
    to a distributed PETSc Vec

    Args:
        v: NumPy array on proc 0, None (or whatever) on other proc

    Returns:
        petsc4py.PETSc.Vec: distributed array on all procs. 
    """

    vloc, x = _distribute_array_helper(v, array_type=PETSc.ScalarType)
    x.setArray(vloc)

    return x


def PETScVec2array(x, rank=0):
    """
    Converts (copies) a distributed PETSc Vec to a sequential array on specified rank

    Args:
        x: PETSc Vec distributed on all procs
        rank: rank receiving the numpy array

    Returns:
        NumPy array on proc 0
    """

    vloc = x.getArray()
    n = x.getSize()

    istart, iend = x.getOwnershipRange()

    nloc = iend - istart
    nloc_loc = MPI_COMM.gather(nloc, root=rank)
    if MPI_RANK != rank:
        nloc_loc = [0]

    istart_loc = MPI_COMM.gather(istart, root=rank)
    if MPI_RANK != rank:
        istart_loc = [0]

    if MPI_RANK == rank:
        v = np_zeros(n, PETSc.ScalarType)
    else:
        v = None

    MPI_COMM.Gatherv(vloc, [v, nloc_loc, istart_loc, _from_numpy_dtype(PETSc.ScalarType)], root=0)

    return v


def PETScMat2coo(A, rank=0):
    """
    Converts a distributed PETSc sparse matrice to a scipy coo
    sparse matrix on a specified rank

    Args:
        A: PETSc Mat distributed on all procs
        rank: rank receiving the coo sparse matrix

    Returns:
        scipy coo sparse matrix on the specified rank
    """

    indptr_loc, indices_loc, data_loc = A.getValuesCSR()
    m, n = A.getSize()

    # gatered values
    iptr_gat = MPI_COMM.gather(indptr_loc, root=rank)
    ind_gat = MPI_COMM.gather(indices_loc, root=rank)
    data_gat = MPI_COMM.gather(data_loc, root=rank)

    if MPI_RANK == rank:
        data = concatenate(data_gat)
        indices = concatenate(ind_gat)

        # reconstruct the indptr array
        indptr = np_zeros(shape=m + 1)
        ind = 1
        offset = 0
        for array in iptr_gat:
            last_elem_ind = len(array) - 2
            for i, elem in enumerate(array[1:]):
                indptr[ind] = offset + elem
                if i == last_elem_ind:
                    offset = indptr[ind]
                ind += 1

        matrix_csr = csr_matrix((data, indices, indptr), shape=(m, n))
        return matrix_csr.tocoo()
    else:
        return None
