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
    "click",
    "cached-property",
    "scipy",
    "numpy",
    "h5py",
    "networkx",
    "morphio",
    "mpi4py",
    "vascpy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "pyyaml",
    "pandas<2.0.0",
    "tables",
    "coverage",
    "libsonata",
    "trimesh",
    "cython",
    "psutil",
]

doc_reqs = [
    "m2r2",
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
    # url="https://bbpteam.epfl.ch/documentation/projects/astrovascpy",
    url="https://github.com/BlueBrain/AstroVascPy",
    # project_urls={
    #     "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/MOLSYS/issues",
    #     "Source": "https://bbpgitlab.epfl.ch/molsys/astrovascpy",
    # },
    project_urls={
        "Tracker": "https://github.com/BlueBrain/AstroVascpy/issues",
        "Source": "https://github.com/BlueBrain/AstroVascPy",
    },
    license="Apache-2",
    packages=find_namespace_packages(include=["astrovascpy*"]),
    python_requires=">=3.10",
    version=VERSION,
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
        "viz": ["vtk"],
    },
    entry_points={
        "console_scripts": [
            "astrovascpy=astrovascpy.cli:main",
        ],
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
