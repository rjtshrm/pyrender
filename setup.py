from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension("pyrender",
              sources=["pyrender.pyx", "render.cpp", "/home/rajat/Desktop/pyrender/glad/src/glad.c"],
              include_dirs=[numpy.get_include(), "/home/rajat/Desktop/pyrender/glad/include/"],
              libraries=["glfw"],
              #extra_compile_args=["-O3", '-std=c++11', '-DON_SCREEN'],
              extra_compile_args=["-O3", '-std=c++11'],
              language="c++")
]

setup(
    ext_modules=cythonize(extensions)
)
