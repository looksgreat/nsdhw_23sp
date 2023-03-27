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
class Matrix {


public:
    Matrix(int rows, int cols) : m_rows(rows), m_cols(cols), m_data(vector<vector<double>>()){
        m_data.resize(m_rows, vector<double>(m_cols, 0));
    }
    Matrix(const Matrix &m) : m_cols(m.cols()), m_rows(m.rows()){
        m_data = m.data();
    }
    double &operator()(int x, int y){
        return m_data[y][x];
    }
    double operator()(int x, int y) const{
        return m_data[y][x];
    }
    bool operator ==(const Matrix &m) const{
        if (m_rows != m.rows() || m_cols != m.cols()){
            return false;
        }
        for (int i = 0; i < m_rows; i++){
            for (int j = 0; j < m_cols; j++){
                if (m_data[i][j] != m(i, j)){
                    return false;
                }
            }
        }
        return true;
    }
    int rows() const{ 
        return m_rows; 
    }
    int cols() const{ 
        return m_cols;
    }
    vector<vector<double>> data() const{
        return m_data;
    }

private:
    int m_rows;
    int m_cols;
    vector<vector<double>> m_data;
};

Matrix multiply_naive(Matrix const &m1, Matrix const &m2){
    Matrix ret(m1.rows(), m2.cols());
    for (int i = 0; i < m1.rows(); i++) {
        for (int j = 0; j < m2.cols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < m1.cols(); k++) {
                sum += m1(i, k) * m2(k, j);
            }
            ret(i, j) = sum;
        }
    }
    return ret;
}

Matrix multiply_tile(Matrix const &m1, Matrix const &m2, int const tile_size){
    int m = m1.rows();
    int n = m2.cols();
    int k = m1.cols();

    Matrix ret(m, n);
    for (int i0 = 0; i0 < m; i0 += tile_size) {
        int imax = std::min(i0 + tile_size, m);
        for (int j0 = 0; j0 < n; j0 += tile_size) {
            int jmax = std::min(j0 + tile_size, n);
            for (int k0 = 0; k0 < k; k0 += tile_size) {
                int kmax = std::min(k0 + tile_size, k);
                for (int i = i0; i < imax; i++) {
                    for (int j = j0; j < jmax; j++) {
                        double sum = 0.0;
                        for (int l = k0; l < kmax; l++) {
                            sum += m1(i, l) * m2(l, j);
                        }
                        ret(i, j) += sum;
                    }
                }
            }
        }
    }

    return ret;
    
}

/*Matrix multiply_mkl(Matrix const &m1, Matrix const &m2){
    mkl_set_num_threads(1);
    Matrix ret(m1.rows(), m2.cols());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.rows(),  m2.cols(), m1.cols(), 1.0 , &m1.data()[0][0],
     m1.cols(), &m2.data()[0][0], m2.cols(), 0.0, &ret.data()[0][0], ret.cols());
    return ret;
    /*Matrix ret(m1.rows(), m2.cols());
    for (int i = 0; i < m1.rows(); i++) {
        for (int j = 0; j < m2.cols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < m1.cols(); k++) {
                sum += m1(i, k) * m2(k, j);
            }
            ret(i, j) = sum;
        }
    }
    return ret;
}*/
Matrix multiply_mkl(Matrix const &mat1, Matrix const &mat2)
{
    if (mat1.cols() != mat2.rows()) {
        exit(1);
    }

    mkl_set_num_threads(1);

    Matrix ret(mat1.rows(), mat2.cols());

    cblas_dgemm(CblasRowMajor /* const CBLAS_LAYOUT Layout */
                ,
                CblasNoTrans /* const CBLAS_TRANSPOSE transa */
                ,
                CblasNoTrans /* const CBLAS_TRANSPOSE transb */
                ,
                mat1.rows() /* const MKL_INT m */
                ,
                mat2.cols() /* const MKL_INT n */
                ,
                mat1.cols() /* const MKL_INT k */
                ,
                1.0 /* const double alpha */
                ,
                &mat1.data()[0][0] /* const double *a */
                ,
                mat1.cols() /* const MKL_INT lda */
                ,
                &mat2.data()[0][0] /* const double *b */
                ,
                mat2.cols() /* const MKL_INT ldb */
                ,
                0.0 /* const double beta */
                ,
                &ret.data()[0][0] /* double * c */
                ,
                ret.cols() /* const MKL_INT ldc */
    );

    return ret;
}

/*PYBIND11_MODULE(_matrix, m){
    m.doc() = "matrix-matrix multiplication";
    m.def("multiply_naive", &multiply_naive, "naive");
    m.def("multiply_tile", &multiply_tile, "tile");
    m.def("multiply_mkl", &multiply_mkl, "mkl");
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int, int>())
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def("__eq__", [](const Matrix &a, const Matrix &b) { 
            return a == b; })
        .def("__setitem__", [](Matrix &self, std::pair<int, int> idx, double val) {
            self(idx.first, idx.second) = val;
        })
        .def("__getitem__", [](const Matrix &self, std::pair<int, int> idx) {
            return self(idx.first, idx.second);
        });
}*/
PYBIND11_MODULE(_matrix, m) {
	// py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
	// .def(py::init<int, int>())
   	// .def_buffer([](Matrix &m) -> py::buffer_info {
    //     return py::buffer_info(
    //         m.data(),                               /* Pointer to buffer */
    //         sizeof(float),                          /* Size of one scalar */
    //         py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
    //         2,                                      /* Number of dimensions */
    //         { m.nrow(), m.ncol() },                 /* Buffer dimensions */
    //         { sizeof(float) * m.ncol(),             /* Strides (in bytes) for each index */
    //           sizeof(float) }
    //     );
    // });
    py::class_<Matrix>(m, "Matrix")
    .def(py::init<int, int>())
    .def("__getitem__", [](Matrix &self, pybind11::args args)
            { 	 
             	//std::cout << (*args[0]) << std::endl;
            	py::tuple t = args[0];
            	// for (size_t it=0; it<t.size(); ++it) {
        		// 	py::print(py::str("{}").format(t[it].cast<int>()));
      			// }
            	return self(t[0].cast<int>(),t[1].cast<int>()); 
         	})
    .def("__setitem__",[](Matrix &self, pybind11::args args)
            { 	 
            	// std::cout << (*args[0]) << " "<< *args[1] << std::endl;
            	py::tuple t = args[0];	
            	self(t[0].cast<int>(),t[1].cast<int>()) = args[1].cast<int>(); 
         	})
    .def_property_readonly("rows", &Matrix::rows)
    .def_property_readonly("cols", &Matrix::cols)
    .def("__eq__", &Matrix::operator ==);

    m.def("multiply_naive", &multiply_naive, "basic Matrix-Matrix Multiplication");
    m.def("multiply_tile", &multiply_tile, "tile Matrix-Matrix Multiplication");
    m.def("multiply_mkl", &multiply_mkl, "mkl Matrix-Matrix Multiplication");
}