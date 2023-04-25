#include<bits/stdc++.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"
#include <mkl/mkl.h>
#include <mkl/mkl_cblas.h>
#include <mkl/mkl_lapack.h>
#include <mkl/mkl_lapacke.h>
using namespace std;
namespace py=pybind11;

template <class T>
class CustomAllocator{
public:
    using value_type = T;
    CustomAllocator() = default;
    
    T* allocate(size_t n){
        if (n > numeric_limits<size_t>::max()/sizeof(T)) throw bad_alloc();
        m_allocated += n*sizeof(T);
        m_byte += n*sizeof(T);
        T *ret = (T *)(malloc(n*sizeof(T)));
        if(ret == nullptr)
            throw std::bad_alloc();
        return ret;
    }
    void deallocate(T *ptr, size_t n){
        m_deallocated += n*sizeof(T);
        m_byte -= n*sizeof(T);
        free(ptr);
    }
    static size_t allocated(){
        return m_allocated;
    }
    static size_t deallocated(){
        return m_deallocated;
    }
    static size_t bytes(){
        return m_byte;
    }
private:
    static size_t m_allocated, m_deallocated , m_byte;
};

template <class T> size_t CustomAllocator<T>::m_allocated = 0;
template <class T> size_t CustomAllocator<T>::m_deallocated = 0;
template <class T> size_t CustomAllocator<T>::m_byte = 0;


class Matrix {
public:
    Matrix(): nrow_(0), ncol_(0) {}
    Matrix(int nrow, int ncol): nrow_(nrow), ncol_(ncol), data_(nrow * ncol) {}

    int nrow() const { return nrow_; }
    int ncol() const { return ncol_; }

    double& operator()(int i, int j) { return data_[i * ncol_ + j]; }
    const double& operator()(int i, int j) const { return data_[i * ncol_ + j]; }
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    bool operator ==(const Matrix &m) const
    {
        if (this->nrow() != m.nrow() || this->ncol() != m.ncol())
            return false;

        for (int i=0; i < this->nrow(); i++)
        {
            for (int j=0; j < this->ncol(); j++)
            {
                if (this->operator()(i, j) != m(i, j))
                    return false;
            }
        }

        return true;
    }

private:
    int nrow_, ncol_;
    std::vector<double, CustomAllocator<double>> data_;
};

Matrix multiply_naive(const Matrix& a, const Matrix& b) {
    if (a.ncol() != b.nrow())
    {
        throw std::out_of_range("matrix column differs from row size");
    }

    Matrix c(a.nrow(), b.ncol());

    for (int i = 0; i < a.nrow(); ++i) {
        for (int j = 0; j < b.ncol(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < a.ncol(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }

    return c;
}

Matrix multiply_tile(const Matrix& a, const Matrix& b, int tile_size) {
    Matrix c(a.nrow(), b.ncol());

    for (int i0 = 0; i0 < a.nrow(); i0 += tile_size) {
        int i1 = std::min(i0 + tile_size, a.nrow());
        for (int j0 = 0; j0 < b.ncol(); j0 += tile_size) {
            int j1 = std::min(j0 + tile_size, b.ncol());
            for (int k0 = 0; k0 < a.ncol(); k0 += tile_size) {
                int k1 = std::min(k0 + tile_size, a.ncol());
                for (int i = i0; i < i1; ++i) {
                    for (int j = j0; j < j1; ++j) {
                        double sum = 0.0;
                        for (int k = k0; k < k1; ++k) {
                            sum += a(i, k) * b(k, j);
                        }
                        c(i, j) += sum;
                    }
                }
            }
        }
    }

    return c;
}

Matrix multiply_mkl(const Matrix& a, const Matrix& b) {
    Matrix c(a.nrow(), b.ncol());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                a.nrow(), b.ncol(), a.ncol(), 1.0,
                a.data(), a.ncol(), b.data(), b.ncol(),
                0.0, const_cast<double*>(c.data()), c.ncol());

    return c;
}


PYBIND11_MODULE(_matrix, m){
    m.doc() = "matrix-matrix multiplication";
    m.def("multiply_naive", &multiply_naive, "naive");
    m.def("multiply_tile", &multiply_tile, "tile");
    m.def("multiply_mkl", &multiply_mkl, "mkl");
    m.def("bytes", &CustomAllocator<double>::bytes);
    m.def("allocated", &CustomAllocator<double>::allocated);
    m.def("deallocated", &CustomAllocator<double>::deallocated);
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def("data", pybind11::overload_cast<>(&Matrix::data))
        .def("data", pybind11::overload_cast<>(&Matrix::data, pybind11::const_))
        .def_property_readonly("nrow", [](const Matrix &mat) { return mat.nrow(); })
        .def_property_readonly("ncol", [](const Matrix &mat) { return mat.ncol(); })
        .def("__eq__", [](const Matrix &a, const Matrix &b) { 
            return a == b; })
        .def("__setitem__", [](Matrix &self, std::pair<int, int> idx, double val) {
            self(idx.first, idx.second) = val;
        })
        .def("__getitem__", [](const Matrix &self, std::pair<int, int> idx) {
            return self(idx.first, idx.second);
        });
}