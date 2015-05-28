#ifndef EBOF_TYPEDEF_H
#define EBOF_TYPEDEF_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

// Typedef EigenMatrices to ensure that they are in row-major order (as required for FFTW)
using RealMatrix 	= Eigen::Matrix<float, 
									Eigen::Dynamic, 
									Eigen::Dynamic, 
									Eigen::RowMajor>;

using ComplexMatrix	= Eigen::Matrix<std::complex<float>, 
									Eigen::Dynamic, 
									Eigen::Dynamic, 
									Eigen::RowMajor>;

using SparseMatrix	= Eigen::SparseMatrix<float,Eigen::RowMajor>;									

#endif // EBOF_TYPEDEF_H