#pragma once

#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <omp.h>

#define EPS 0.0001

template <typename T>
class Matrix {
public:
    Matrix() = default;
    Matrix(size_t rows, size_t cols);
    explicit Matrix(std::vector<std::vector<T>> v);

    T det() const;
    void transpose();
    Matrix<T> inverseEMethod() const;
    Matrix<T> inverseEMethod_() const; //multithreading realization
    std::ostream& print(std::ostream& out);

    size_t rows_size() const;
    size_t cols_size() const;

    const std::vector<T>& operator[](size_t row) const;
    std::vector<T>& operator[](size_t row);

private:
    std::vector<std::vector<T>> A;
};

template <typename T>
Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B);

template <typename T>
void operator*(const T& m, Matrix<T>& A);

template <typename T>
Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B);

template <typename T>
bool operator==(const Matrix<T>& A, const Matrix<T>& B);




//-----------------------------------
template <typename T>
T Matrix<T>::det() const {
    if(A.size() != A[0].size())
        throw std::domain_error("Impossible to calculate the determinant for a non-square matrix");

    T det = 1;

    Matrix tmp = Matrix(A);

    for (size_t i = 0; i < tmp.rows_size(); ++i) {
        size_t k = i;
        for (size_t j = i + 1; j < tmp.rows_size(); ++j)
            if (abs (tmp[j][i]) > abs (tmp[k][i]))
                k = j;
        if (abs (tmp[k][i]) < EPS) {
            det = 0;
            break;
        }
        std::swap(tmp[i], tmp[k]);
        if (i != k)
            det = -det;
        det *= tmp[i][i];
        for (size_t j = i + 1; j < tmp.rows_size(); ++j)
            tmp[i][j] /= tmp[i][i];
        for (size_t j = 0; j < tmp.rows_size(); ++j)
            if (j != i && abs (tmp[j][i]) > EPS)
                for (size_t m = i + 1; m < tmp.rows_size(); ++m)
                    tmp[j][m] -= tmp[i][m] * tmp[j][i];
    }

    return det;
}

template <typename T>
void Matrix<T>::transpose() {
    std::vector<std::vector<T>> Tr(A[0].size(), std::vector<T>(A.size()));

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size; ++j) {
            Tr[j][i] = A[i][j];
        }
    }

    A = Tr;
}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) {
    Matrix();
    A.resize(rows, std::vector<T>(cols));
}

template<typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> v)
    : A(std::move(v))
{}

template<typename T>
std::ostream& Matrix<T>::print(std::ostream& out) {
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            if(A[i][j] < EPS)
                out << 0;
            else
                out << A[i][j];

            if(j != A[0].size() - 1)
                out << ' ';
        }
        out << '\n';
    }
    out << '\n';

    return out;
}

template<typename T>
size_t Matrix<T>::rows_size() const {
    return A.size();
}

template<typename T>
size_t Matrix<T>::cols_size() const {
    return A[0].size();
}

template<typename T>
const std::vector<T>& Matrix<T>::operator[](size_t row) const {
    return A[row];
}

template<typename T>
std::vector<T> &Matrix<T>::operator[](size_t row) {
    return A[row];
}

template<typename T>
Matrix<T> Matrix<T>::inverseEMethod() const {
    T temp;
    Matrix tmp = Matrix(A);
    std::vector<std::vector<T>> E(A.size(), std::vector<T>(A.size(), 0));
    for(int64_t i = 0; i < A.size(); ++i) E[i][i] = 1;

    for (int64_t k = 0; k < A.size(); k++) {
        temp = tmp[k][k];

        for (int64_t j = 0; j < A.size(); j++) {
            tmp[k][j] /= temp;
            E[k][j] /= temp;
        }

        for (int64_t i = k + 1; i < A.size(); i++) {
            temp = tmp[i][k];

            for (int64_t j = 0; j < A.size(); j++) {
                tmp[i][j] -= tmp[k][j] * temp;
                E[i][j] -= E[k][j] * temp;
            }
        }
    }

    for (int64_t k = A.size() - 1; k > 0; k--) {
        for (int64_t i = k - 1; i >= 0; i--) {
            temp = tmp[i][k];

            for (int64_t j = 0; j < A.size(); j++) {
                tmp[i][j] -= tmp[k][j] * temp;
                E[i][j] -= E[k][j] * temp;
            }
        }
    }

    for (int64_t i = 0; i < A.size(); i++) {
        for (int64_t j = 0; j < A.size(); j++) {
            tmp[i][j] = E[i][j];
        }
    }

    return tmp;
}

template<typename T>
Matrix<T> Matrix<T>::inverseEMethod_() const {
    T temp;
    Matrix tmp = Matrix(A);
    std::vector<std::vector<T>> E(A.size(), std::vector<T>(A.size(), 0));

    #pragma omp parallel for
    for(int64_t i = 0; i < A.size(); ++i) E[i][i] = 1;

    #pragma omp parallel for
    for (int64_t k = 0; k < A.size(); k++) {
        temp = tmp[k][k];

        for (int64_t j = 0; j < A.size(); j++) {
            tmp[k][j] /= temp;
            E[k][j] /= temp;
        }

        for (int64_t i = k + 1; i < A.size(); i++) {
            temp = tmp[i][k];

            for (int64_t j = 0; j < A.size(); j++) {
                tmp[i][j] -= tmp[k][j] * temp;
                E[i][j] -= E[k][j] * temp;
            }
        }
    }

    for (int64_t k = A.size() - 1; k > 0; k--) {
        for (int64_t i = k - 1; i >= 0; i--) {
            temp = tmp[i][k];

            #pragma omp parallel for
            for (int64_t  j = 0; j < A.size(); j++) {
                tmp[i][j] -= tmp[k][j] * temp;
                E[i][j] -= E[k][j] * temp;
            }
        }
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < A.size(); i++) {
        for (int64_t j = 0; j < A.size(); j++) {
            tmp[i][j] = E[i][j];
        }
    }

    return tmp;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
    if(A.cols_size() != B.rows_size())
        throw std::invalid_argument("Impossible multiple matrices with different number first_columns and second_rows");

    std::vector<std::vector<T>> M(A.rows_size(), std::vector<T>(B.cols_size()));

    for (int i = 0; i < A.rows_size(); i++) {
        for (int j = 0; j < B.cols_size(); j++) {
            M[i][j] = 0;
            for (int k = 0; k < A.cols_size(); k++)
                M[i][j] += A[i][k] * B[k][j];
        }
    }

    return Matrix(M);
}

template <typename T>
void operator*(const T& m, Matrix<T>& A) {
    for (int i = 0; i < A.rows_size(); ++i) {
        for (int j = 0; j < A.cols_size(); ++j) {
            A[i][j] *= m;
        }
    }
}

template <typename T>
Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B) {
    if(A.rows_size() != B.rows_size())
        throw std::invalid_argument("Impossible add matrices with different number of rows");

    if(A.cols_size() != B.cols_size())
        throw std::invalid_argument("Impossible add matrices with different number of columns");

    std::vector<std::vector<T>> S(A.rows_size(), std::vector<T>(B.cols_size()));
    for (int i = 0; i < A.rows_size(); ++i) {
        for (int j = 0; j < A.cols_size(); ++j) {
            S[i][j] = A[i][j] + B[i][j];
        }
    }

    return Matrix(S);
}

template <typename T>
bool operator==(const Matrix<T>& A, const Matrix<T>& B) {
    if(A.rows_size() != B.rows_size() || A.cols_size() != B.cols_size())
        return false;

    for (int i = 0; i < A.rows_size(); ++i) {
        for (int j = 0; j < A.cols_size(); ++j) {
            if(abs(A[i][j] - B[i][j]) > EPS)
                return false;
        }
    }

    return true;
}
