# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'build.py.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/29 0:29'
"""
from setuptools import setup, Extension
import pybind11
import re

source_file = "export.cpp"

def get_module_name():
    text = open(source_file, "r").read()
    module_name = re.search(r"(?<=PYBIND11_MODULE\()(\w+)(?=,)",text).group()
    return module_name


cpp_args = ['/O2', '/EHsc', '/arch:x64']
module_name = get_module_name()
functions_module = Extension(
    name=module_name,
    sources=[source_file],
    language='c++',
    include_dirs=[pybind11.get_include()],
    extra_compile_args=cpp_args,
)
if __name__ == "__main__":

    pass
    setup(name=module_name,
    version='1.0',
    description='Python package with superfastcode2 C++ extension (PyBind11)',ext_modules=[functions_module])