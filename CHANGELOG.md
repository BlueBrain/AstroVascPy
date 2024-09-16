# Changelog

### v0.1.6
* New parameter `entry_noise` to enable or disable the endfeet activity on entry nodes. (#41)
* Add helper script to load archngv graphs and convert them in pickle binary format. (#40)

### v0.1.5
* Lazy import of mpi4py module (#27)

### v0.1.4
* Bump minimal supported version of Python to 3.10. Continuous integration now uses Python 3.11 (#23)

### v0.1.3
* New function `distribute_array` for scattering numpy arrays. (#17)
* PETSc solver: replaced GMRES with LGMRES. Added null space information. (#20)

### v0.1.2
* Introduce the class `utils.Graph` to optimize the computation of node degrees. (#12)
* Compute flow and pressure only on the main connected component. (#12)

### v0.1.1
* vkt is now an optional dependency. Use `pip install astrovascpy[viz]` to enable it. (#14)
* Fix PetscBinaryIO.get_conf() value on error. Returns a valid config when PETSc installation cannot be located. (#14)

### v0.1
* Add unit tests for MPI-PETSc functions. (#5)
* Add separate ROU calibration for arteries and capillaries. (#6)
* Remove complex number support from petsc & petsc4py
* Initial release of AstroVascPy Python library.
