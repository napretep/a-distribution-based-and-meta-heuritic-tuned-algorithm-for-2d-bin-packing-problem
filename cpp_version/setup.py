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
import re,os
from datetime import datetime
source_file = "setup.cpp"

ROOT_PATH = os.path.split(os.path.abspath(__file__))[0]


def get_module_name():
    all_cpp_path = os.path.join(ROOT_PATH,"cpp-solver","all.cpp")
    export_path =  os.path.join(ROOT_PATH,"export.cpp")

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
    include_dirs=[pybind11.get_include(),os.path.join(ROOT_PATH,"cpp-solver","eigen-3.4.0")],
    extra_compile_args=cpp_args,
)

if __name__ == "__main__":
    # 创建一个datetime对象，表示某个特定的日期和时间
    dt = datetime.now() # 年，月，日，时，分，秒

    # 获取年份
    year = dt.year-2000

    # 获取该日期在当年的秒数
    seconds =int((dt - datetime(dt.year, 1, 1)).total_seconds())

    setup(name=module_name,
    version=f"{year}.{seconds}",
    description='new dist3 algo',ext_modules=[functions_module])