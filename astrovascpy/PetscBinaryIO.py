"""
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

PetscBinaryIO
===============
COPIED FROM PETSC Library -> $PETSC_DIR/lib/petsc/bin (along with get_conf function)
===============

Since PetscBinaryIO is not part of the PETSC core library, but rather a showcase Python script,
we decided to copy it here, and avoid setup tricks like:
export PYTHONPATH=$PETSC_DIR/lib/petsc/bin:$PYTHONPATH
Provides
  1. PETSc-named objects Vec, Mat, and IS that inherit numpy.ndarray
  2. A class to read and write these objects from PETSc binary files.
The standard usage of this module should look like:
  >>> import PetscBinaryIO
  >>> io = PetscBinaryIO.PetscBinaryIO()
  >>> objects = io.readBinaryFile('file.dat')
or
  >>> import PetscBinaryIO
  >>> import numpy
  >>> vec = numpy.array([1., 2., 3.]).view(PetscBinaryIO.Vec)
  >>> io = PetscBinaryIO.PetscBinaryIO()
  >>> io.writeBinaryFile('file.dat', [vec,])
See also PetscBinaryIO.__doc__ and methods therein.
"""

import functools
import importlib.metadata
import os
import warnings

import numpy as np
from scipy.sparse import csr_matrix


def get_conf():
    """Parses various PETSc configuration/include files to get data types.

    precision, indices, complexscalars = get_conf()

    Output:
      precision: 'single', 'double', 'longlong' indicates precision of PetscScalar
      indices: '32', '64' indicates bit-size of PetscInt
      complex: True/False indicates whether PetscScalar is complex or not.
    """

    precision = None
    indices = None
    complexscalars = None
    DEFAULT_CONF = ("double", "64bit", False)

    if "PETSC_DIR" in os.environ:
        petscdir = os.environ["PETSC_DIR"]
    else:
        try:
            petsc = importlib.metadata.distribution("petsc")
        except importlib.metadata.PackageNotFoundError:
            try:
                petsc4py = importlib.metadata.distribution("petsc4py")
                if "conda" in str(petsc4py.files[0].locate()):
                    petscdir = os.environ["CONDA_PREFIX"]
                else:
                    raise importlib.metadata.PackageNotFoundError
            except importlib.metadata.PackageNotFoundError:
                warnings.warn("Unable to locate PETSc installation, using defaults")
                return DEFAULT_CONF
        else:
            petscdir = str(petsc._path.parent / "petsc")

    if os.path.isfile(os.path.join(petscdir, "lib", "petsc", "conf", "petscrules")):
        # found prefix install
        petscvariables = os.path.join(petscdir, "lib", "petsc", "conf", "petscvariables")
        petscconfinclude = os.path.join(petscdir, "include", "petscconf.h")
    else:
        if "PETSC_ARCH" in os.environ:
            petscarch = os.environ["PETSC_ARCH"]
            if os.path.isfile(
                os.path.join(petscdir, petscarch, "lib", "petsc", "conf", "petscrules")
            ):
                # found legacy install
                petscvariables = os.path.join(
                    petscdir, petscarch, "lib", "petsc", "conf", "petscvariables"
                )
                petscconfinclude = os.path.join(petscdir, petscarch, "include", "petscconf.h")
            else:
                warnings.warn(
                    "Unable to locate PETSc installation in specified PETSC_DIR/PETSC_ARCH, \
                     using defaults"
                )
                return DEFAULT_CONF
        else:
            warnings.warn(
                "PETSC_ARCH env not set or incorrect PETSC_DIR is given - \
                 unable to locate PETSc installation, using defaults"
            )
            return DEFAULT_CONF

    try:
        fid = open(petscvariables, "r")
    except IOError:
        warnings.warn("Nonexistent or invalid PETSc installation, using defaults")
        return DEFAULT_CONF
    else:
        for line in fid:
            if line.startswith("PETSC_PRECISION"):
                precision = line.strip().split("=")[1].strip("\n").strip()

        fid.close()

    try:
        fid = open(petscconfinclude, "r")
    except IOError:
        warnings.warn("Nonexistent or invalid PETSc installation, using defaults")
        return DEFAULT_CONF
    else:
        for line in fid:
            if line.startswith("#define PETSC_USE_64BIT_INDICES 1"):
                indices = "64bit"
            elif line.startswith("#define PETSC_USE_COMPLEX 1"):
                complexscalars = True

        if indices is None:
            indices = "32bit"
        if complexscalars is None:
            complexscalars = False
        fid.close()

    return precision, indices, complexscalars


def update_wrapper_with_doc(wrapper, wrapped):
    """Similar to functools.update_wrapper, but also gets the wrapper's __doc__ string"""
    wdoc = wrapper.__doc__

    functools.update_wrapper(wrapper, wrapped)
    if wdoc is not None:
        if wrapper.__doc__ is None:
            wrapper.__doc__ = wdoc
        else:
            wrapper.__doc__ = wrapper.__doc__ + wdoc
    return wrapper


def wraps_with_doc(wrapped):
    """Similar to functools.wraps, but also gets the wrapper's __doc__ string"""
    return functools.partial(update_wrapper_with_doc, wrapped=wrapped)


def decorate_with_conf(f):
    """Decorates methods to take kwargs for precisions."""

    @wraps_with_doc(f)
    def decorated_f(self, *args, **kwargs):
        """
        Additional kwargs:
          precision: 'single', 'double', 'longlong' for scalars
          indices: '32bit', '64bit' integer size
          complexscalars: True/False

          Note these are set in order of preference:
            1. kwargs if given here
            2. PetscBinaryIO class __init__ arguments
            3. PETSC_DIR/PETSC_ARCH defaults
        """

        changed = False
        old_precision = self.precision
        old_indices = self.indices
        old_complexscalars = self.complexscalars

        try:
            self.precision = kwargs.pop("precision")
        except KeyError:
            pass
        else:
            changed = True

        try:
            self.indices = kwargs.pop("indices")
        except KeyError:
            pass
        else:
            changed = True

        try:
            self.complexscalars = kwargs.pop("complexscalars")
        except KeyError:
            pass
        else:
            changed = True

        if changed:
            self._update_dtypes()

        result = f(self, *args, **kwargs)

        if changed:
            self.precision = old_precision
            self.indices = old_indices
            self.complexscalars = old_complexscalars
            self._update_dtypes()

        return result

    return decorated_f


class DoneWithFile(Exception):
    pass


class Vec(np.ndarray):
    """Vec represented as 1D numpy array

    The best way to instantiate this class for use with writeBinaryFile()
    is through the numpy view method:

    vec = numpy.array([1,2,3]).view(Vec)
    """

    _classid = 1211214


class MatDense(np.matrix):
    """Mat represented as 2D numpy array

    The best way to instantiate this class for use with writeBinaryFile()
    is through the numpy view method:

    mat = numpy.array([[1,0],[0,1]]).view(Mat)
    """

    _classid = 1211216


class MatSparse(tuple):
    """Mat represented as CSR tuple ((M, N), (rowindices, col, val))

    This should be instantiated from a tuple:

    mat = MatSparse( ((M,N), (rowindices,col,val)) )
    """

    _classid = 1211216

    def __repr__(self):
        return "MatSparse: %s" % super(MatSparse, self).__repr__()


class IS(np.ndarray):
    """IS represented as 1D numpy array

    The best way to instantiate this class for use with writeBinaryFile()
    is through the numpy "view" method:

    an_is = numpy.array([3,4,5]).view(IS)
    """

    _classid = 1211218


class PetscBinaryIO(object):
    """Reader/Writer class for PETSc binary files.

    Note that by default, precisions for both scalars and indices, as well as
    complex scalars, are picked up from the PETSC_DIR/PETSC_ARCH configuration
    as set by environmental variables.

    Alternatively, defaults can be overridden at class instantiation, or for
    a given method call.
    """

    _classid = {1211216: "Mat", 1211214: "Vec", 1211218: "IS", 1211219: "Bag"}

    def __init__(self, precision=None, indices=None, complexscalars=None):
        if (precision is None) or (indices is None) or (complexscalars is None):
            defaultprecision, defaultindices, defaultcomplexscalars = get_conf()
            if precision is None:
                if defaultprecision is None:
                    precision = "double"
                else:
                    precision = defaultprecision

            if indices is None:
                if defaultindices is None:
                    indices = "32bit"
                else:
                    indices = defaultindices

            if complexscalars is None:
                if defaultcomplexscalars is None:
                    complexscalars = False
                else:
                    complexscalars = defaultcomplexscalars

        self.precision = precision
        self.indices = indices
        self.complexscalars = complexscalars
        self._update_dtypes()

    def _update_dtypes(self):
        if self.indices == "64bit":
            self._inttype = np.dtype(">i8")
        else:
            self._inttype = np.dtype(">i4")

        if self.precision == "longlong":
            nbyte = 16
        elif self.precision == "single":
            nbyte = 4
        else:
            nbyte = 8

        if self.complexscalars:
            name = "c"
            nbyte = nbyte * 2  # complex scalar takes twice as many bytes
        else:
            name = "f"

        self._scalartype = ">{0}{1}".format(name, nbyte)

    @decorate_with_conf
    def readVec(self, fh):
        """Reads a PETSc Vec from a binary file handle, returning just the data."""

        nz = np.fromfile(fh, dtype=self._inttype, count=1)[0]
        try:
            vals = np.fromfile(fh, dtype=self._scalartype, count=nz)
        except MemoryError:
            raise IOError("Inconsistent or invalid Vec data in file")
        if len(vals) == 0:
            raise IOError("Inconsistent or invalid Vec data in file")
        return vals.view(Vec)

    @decorate_with_conf
    def writeVec(self, fh, vec):
        """Writes a PETSc Vec to a binary file handle."""

        metadata = np.array([Vec._classid, len(vec)], dtype=self._inttype)
        metadata.tofile(fh)
        vec.astype(self._scalartype).tofile(fh)
        return

    @decorate_with_conf
    def readMatSparse(self, fh):
        """Reads a PETSc Mat, returning a sparse representation of the data.

        (M,N), (I,J,V) = readMatSparse(fid)

        Input:
          fid : file handle to open binary file.
        Output:
          M,N : matrix size
          I,J : arrays of row and column for each nonzero
          V: nonzero value
        """

        try:
            M, N, nz = np.fromfile(fh, dtype=self._inttype, count=3)
            I = np.empty(M + 1, dtype=self._inttype)  # noqa: E741
            I[0] = 0
            rownz = np.fromfile(fh, dtype=self._inttype, count=M)
            np.cumsum(rownz, out=I[1:])
            assert I[-1] == nz

            J = np.fromfile(fh, dtype=self._inttype, count=nz)
            assert len(J) == nz
            V = np.fromfile(fh, dtype=self._scalartype, count=nz)
            assert len(V) == nz
        except (AssertionError, MemoryError, IndexError):
            raise IOError("Inconsistent or invalid Mat data in file")

        return MatSparse(((M, N), (I, J, V)))

    @decorate_with_conf
    def writeMatSparse(self, fh, mat):
        """Writes a Mat into a PETSc binary file handle"""

        ((M, N), (I, J, V)) = mat  # noqa: E741
        metadata = np.array([MatSparse._classid, M, N, I[-1]], dtype=self._inttype)
        rownz = I[1:] - I[:-1]

        assert len(J.shape) == len(V.shape) == len(I.shape) == 1
        assert len(J) == len(V) == I[-1] == rownz.sum()
        assert (rownz > -1).all()

        metadata.tofile(fh)
        rownz.astype(self._inttype).tofile(fh)
        J.astype(self._inttype).tofile(fh)
        V.astype(self._scalartype).tofile(fh)
        return

    @decorate_with_conf
    def readMatDense(self, fh):
        """Reads a PETSc Mat, returning a dense representation of the data."""

        try:
            M, N, nz = np.fromfile(fh, dtype=self._inttype, count=3)
            I = np.empty(M + 1, dtype=self._inttype)  # noqa: E741
            I[0] = 0
            rownz = np.fromfile(fh, dtype=self._inttype, count=M)
            np.cumsum(rownz, out=I[1:])
            assert I[-1] == nz

            J = np.fromfile(fh, dtype=self._inttype, count=nz)
            assert len(J) == nz
            V = np.fromfile(fh, dtype=self._scalartype, count=nz)
            assert len(V) == nz

        except (AssertionError, MemoryError, IndexError):
            raise IOError("Inconsistent or invalid Mat data in file")

        mat = np.zeros((M, N), dtype=self._scalartype)
        for row in range(M):
            rstart, rend = I[row : row + 2]
            mat[row, J[rstart:rend]] = V[rstart:rend]
        return mat.view(MatDense)

    @decorate_with_conf
    def readMatSciPy(self, fh):
        (M, N), (I, J, V) = self.readMatSparse(fh)  # noqa: E741
        return csr_matrix((V, J, I), shape=(M, N))

    @decorate_with_conf
    def writeMatSciPy(self, fh, mat):
        if hasattr(mat, "tocsr"):
            mat = mat.tocsr()
        assert isinstance(mat, csr_matrix)
        V = mat.data  # noqa: F841
        M, N = mat.shape
        J = mat.indices  # noqa: F841
        I = mat.indptr  # noqa: E741,F841
        return self.writeMatSparse(fh, (mat.shape, (mat.indptr, mat.indices, mat.data)))

    @decorate_with_conf
    def readMat(self, fh, mattype="sparse"):
        """Reads a PETSc Mat from binary file handle.

        optional mattype: 'sparse" or 'dense'

        See also: readMatSparse, readMatDense
        """

        if mattype == "sparse":
            return self.readMatSparse(fh)
        elif mattype == "dense":
            return self.readMatDense(fh)
        elif mattype == "scipy.sparse":
            return self.readMatSciPy(fh)
        else:
            raise RuntimeError("Invalid matrix type requested: choose sparse/dense")

    @decorate_with_conf
    def readIS(self, fh):
        """Reads a PETSc Index Set from binary file handle."""

        try:
            nz = np.fromfile(fh, dtype=self._inttype, count=1)[0]
            v = np.fromfile(fh, dtype=self._inttype, count=nz)
            assert len(v) == nz
        except (MemoryError, IndexError):
            raise IOError("Inconsistent or invalid IS data in file")
        return v.view(IS)

    @decorate_with_conf
    def writeIS(self, fh, anis):
        """Writes a PETSc IS to binary file handle."""

        metadata = np.array([IS._classid, len(anis)], dtype=self._inttype)
        metadata.tofile(fh)
        anis.astype(self._inttype).tofile(fh)
        return

    @decorate_with_conf
    def readBinaryFile(self, fid, mattype="sparse"):
        """Reads a PETSc binary file, returning a tuple of the contained objects.

        objects = self.readBinaryFile(fid, **kwargs)

        Input:
          fid : either file name or handle to an open binary file.

        Output:
          objects : tuple of objects representing the data in numpy arrays.

        Optional:
          mattype :
            'sparse': Return matrices as raw CSR: (M, N), (row, col, val).
            'dense': Return matrices as MxN numpy arrays.
            'scipy.sparse': Return matrices as scipy.sparse objects.
        """

        close = False

        if isinstance(fid, str):
            fid = open(fid, "rb")
            close = True

        objects = []
        try:
            while True:
                # read header
                try:
                    header = np.fromfile(fid, dtype=self._inttype, count=1)[0]
                except (MemoryError, IndexError):
                    raise DoneWithFile
                try:
                    objecttype = self._classid[header]
                except KeyError:
                    raise IOError(
                        "Invalid PetscObject CLASSID or object not implemented for python"
                    )

                if objecttype == "Vec":
                    objects.append(self.readVec(fid))
                elif objecttype == "IS":
                    objects.append(self.readIS(fid))
                elif objecttype == "Mat":
                    objects.append(self.readMat(fid, mattype))
                elif objecttype == "Bag":
                    raise NotImplementedError("Bag Reader not yet implemented")
        except DoneWithFile:
            pass
        finally:
            if close:
                fid.close()

        return tuple(objects)

    @decorate_with_conf
    def writeBinaryFile(self, fid, objects):
        """Writes a PETSc binary file containing the objects given.

        readBinaryFile(fid, objects)

        Input:
          fid : either file handle to an open binary file, or filename.
          objects : list of objects representing the data in numpy arrays,
                    which must be of type Vec, IS, MatSparse, or MatSciPy.
        """
        close = False
        if isinstance(fid, str):
            fid = open(fid, "wb")
            close = True

        for petscobj in objects:
            if isinstance(petscobj, Vec):
                self.writeVec(fid, petscobj)
            elif isinstance(petscobj, IS):
                self.writeIS(fid, petscobj)
            elif isinstance(petscobj, MatSparse):
                self.writeMatSparse(fid, petscobj)
            elif isinstance(petscobj, MatDense):
                if close:
                    fid.close()
                raise NotImplementedError("Writing a dense matrix is not yet supported")
            else:
                try:
                    self.writeMatSciPy(fid, petscobj)
                except AssertionError:
                    if close:
                        fid.close()
                    raise TypeError("Object %s is not a valid PETSc object" % (petscobj.__repr__()))
        if close:
            fid.close()
        return
