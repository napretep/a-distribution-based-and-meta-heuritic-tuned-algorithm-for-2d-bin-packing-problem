#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;


PYBIND11_MODULE(helloWorld, m) {
    m.def("add", [](int a, int b) { return 2*a + b; }, "A function which adds two numbers");
}
