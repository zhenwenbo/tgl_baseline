#include <pybind11/pybind11.h>
#include <vector>
#include <algorithm> // 包含这个头文件以使用std::sort
#include <pybind11/stl.h>
#include <omp.h>
#include <execution>


void sort_function(std::vector<int>& vec) {
    std::sort(std::execution::par, vec.begin(), vec.end());
}


void parallel_quick_sort(std::vector<int>& vec, int low, int high) {
    if (low < high) {
        int pivot = vec[low];
        int i = low, j = high;

        while (i < j) {
            while (vec[i] < pivot) i++;
            while (vec[j] > pivot) j--;

            if (i <= j) {
                std::swap(vec[i], vec[j]);
                i++;
                j--;
            }
        }

        #pragma omp task shared(vec)
        {
            parallel_quick_sort(vec, low, j);
        }

        #pragma omp task shared(vec)
        {
            parallel_quick_sort(vec, i, high);
        }

        #pragma omp taskwait
    }
}

void parallel_sort(std::vector<int>& vec) {
    #pragma omp parallel
    {
        parallel_quick_sort(vec, 0, vec.size() - 1);
    }
}

PYBIND11_MODULE(cpp_sort, m) {
    m.def("sort", &sort_function, "Sorts a vector of integers")
    .def("mp_sort", &parallel_sort);
}