#!/usr/bin/env python3
"""
dRoute build setup for the C++ extension.

Requires:
    - CMake 3.15+
    - C++17 compiler
    - pybind11 (auto-downloaded if not found)
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building with cmake."""
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using cmake."""

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DDMC_BUILD_PYTHON=ON",
            "-DDMC_BUILD_TESTS=OFF",
        ]

        # Add any extra CMAKE_ARGS from environment
        if "CMAKE_ARGS" in os.environ:
            import shlex
            cmake_args += shlex.split(os.environ["CMAKE_ARGS"])

        build_args = ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Default to a sensible number of parallel jobs
            jobs = os.cpu_count() or 4
            build_args += ["--", f"-j{jobs}"]

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True
        )

        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True
        )


setup(
    name="droute",
    # Keep in sync with python/droute/_version.py (the source of truth pyproject reads dynamically).
    version="0.6.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("_droute_core")],
    cmdclass={"build_ext": CMakeBuild},
)
