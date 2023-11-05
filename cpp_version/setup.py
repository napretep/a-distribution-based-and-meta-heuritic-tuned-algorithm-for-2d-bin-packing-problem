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

source_file = "setup.cpp"

def get_module_name():
    all_cpp_path = "./cpp-solver/all.cpp"
    export_path = "./export.cpp"
    all_cpp_code = open(all_cpp_path, "r").read()
    export_code = open(export_path, "r").read()
    compile_code = all_cpp_code + "\n" + export_code
    open(source_file, "w").write(compile_code)

    text = open(source_file, "r").read()
    module_name = re.search(r"(?<=PYBIND11_MODULE\()(\w+)(?=,)",text).group()
    return module_name


cpp_args = ['/O2', '/EHsc', '/std:c++20']

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