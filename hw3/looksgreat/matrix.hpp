#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
using namespace std;
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

#endif /* MATRIX_HPP */
