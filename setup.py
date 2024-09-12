"""Setup for the astrovascpy package."""

import importlib.util
from pathlib import Path

from setuptools import find_namespace_packages, setup

spec = importlib.util.spec_from_file_location(
    "astrovascpy.version",
    "astrovascpy/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

reqs = [
    "cached-property",
    "click",
    "coverage",
    "cython",
    "h5py",
    "libsonata",
    "matplotlib",
    "morphio",
    "mpi4py",
    "networkx",
    "numpy",
    "pandas",
    "psutil",
    "pyyaml",
    "scipy",
    "seaborn",
    "tables",
    "tqdm",
    "trimesh",
    "vascpy",
]

doc_reqs = [
    "sphinx-mdinclude",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-click",
]

test_reqs = [
    "pytest",
    "pytest-mpi",
]

setup(
    name="AstroVascPy",
    description="Simulating blood flow in vasculature",
    author="Blue Brain Project, EPFL",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/BlueBrain/AstroVascPy",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/AstroVascpy/issues",
        "Source": "https://github.com/BlueBrain/AstroVascPy",
    },
    license="Apache-2",
    packages=find_namespace_packages(include=["astrovascpy*"]),
    python_requires=">=3.11",
    version=VERSION,
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
        "viz": ["vtk"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
