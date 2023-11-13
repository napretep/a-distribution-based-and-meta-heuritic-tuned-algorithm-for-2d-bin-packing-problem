//-------------------pybind11 zone---------------------------------------------


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;



std::unique_ptr<Algo>  single_run(py::array_t<float> input_array, pair<float, float>material, std::optional<vector<float>> parameter_input_array=std::nullopt,string algo_type="Dist",bool is_debug=false) {
    py::buffer_info buf_info = input_array.request();

    float *ptr = static_cast<float *>(buf_info.ptr);
    int num_elements = buf_info.size;
    std::vector<float> items(ptr, ptr + num_elements);
    return inner_single_run(items,material,parameter_input_array,algo_type,is_debug);
}

vector<float> multi_run(
    py::array_t<float>input_array, 
    pair<float, float>material, 
    std::optional<vector<float>> parameter_input_array=std::nullopt,
    string algo_type="Dist", int run_count=27,bool need_report=false) {

    py::buffer_info buf_info = input_array.request();
    float *ptr = static_cast<float *>(buf_info.ptr);
    int num_elements = buf_info.size;
    vector<float> pre_items(ptr, ptr + num_elements);
    vector<vector<float>> item_queue;
    int data_size = pre_items.size()/run_count;
    for(auto i=0;i<pre_items.size();i++){
        if (i%data_size==0){
            vector<float>new_dataset;
            item_queue.push_back(new_dataset);
        }
        item_queue.back().push_back(pre_items[i]);
    }
    vector<float> results;
    for(auto item: item_queue){
        std::unique_ptr<Algo> algo = inner_single_run(item, material, parameter_input_array, algo_type);
        float r = algo->get_avg_util_rate();
        if (need_report) {
            cout << algo->task_id << " done";
        }
        results.push_back(r);
    }
    return results;
}

//get_algo_parameters_length
PYBIND11_MODULE(BinPacking2DAlgo, m) {
    py::class_<Algo>(m, "Algo")
        .def("get_avg_util_rate", &Algo::get_avg_util_rate)
        .def("solution_as_vector",&Algo::solution_as_vector, "return:list:solution[list:each-plan[list:item|container[ax,ay,bx,by]]]")
        .def_readonly("packinglog", &Algo::packinglog, "return:list:solution[list:history-plan[list:each-plan[list:item|container[ax,ay,bx,by]]]]")
        .def_readonly("task_id", &Algo::task_id);
    m.def("single_run", &single_run,
            py::arg("input_array"), 
            py::arg("material"),
            py::arg("parameter_input_array")=std::nullopt,
            py::arg("algo_type")="Dist",
            py::arg("is_debug") = false,
            "algo_type=Dist,MaxRect,Skyline,Dist_MaxRect,Dist_Skyline. Dist and Dist2 need learn parameters to performance well"
            );
    m.def("multi_run",&multi_run,
            py::arg("input_array"), 
            py::arg("material"),
            py::arg("parameter_input_array")=std::nullopt,
            py::arg("algo_type")="Dist", 
            py::arg("run_count")=27,
            py::arg("need_report")=false,
            "algo_type=Dist,MaxRect,Skyline,Dist_MaxRect,Dist_Skyline. Dist and Dist2 need learn parameters to performance well"
    );
    m.def("get_algo_parameters_length",&get_algo_parameters_length,
            py::arg("algo_type"),
            "algo_type=Dist,MaxRect,Skyline,Dist_MaxRect,Dist_Skyline. Dist and Dist2 need learn parameters to performance well"
    );

}
