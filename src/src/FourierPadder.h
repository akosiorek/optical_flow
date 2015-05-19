#ifndef FOURIER_PADDER_H
#define FOURIER_PADDER_H

#include <memory>
#include <stdint.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// Determines the next power of two at run time
template<uint32_t A, uint8_t B = 16>
struct Pow2RoundUp { enum{ value = Pow2RoundUp<((B == 16 ? (A-1) : A) | ((B == 16 ? (A-1) : A) >> B)), B/2>::value }; };
template<uint32_t A >
struct Pow2RoundUp<A, 1> { enum{ value = ((A | (A >> 1)) + 1) }; };

/**
 * @brief Constexpr function to determine the max of two values at compile time
 * 
 * @param a First value
 * @param b Second value
 * 
 * @return Returns the larger element by value
 */
template<typename T> constexpr
T const& max(T const& a, T const& b)
{
	return a > b ? a : b;
}

/**
 * @brief This class zero-pads matrices for efficient FFT.
 * @details Matrix size typically needs to be a power of two to be efficiently FFT'ed.
 * 			(Actually, e.g., prime numbers etc. might do it as well, but this is really just for now.)
 * 
 * @tparam int dataSize Size of data matrix
 * @tparam int filterSize Size of filter
 */
template<unsigned int dataSize, unsigned int filterSize>
class FourierPadder
{
public:
	using Ptr = std::shared_ptr<FourierPadder>;

	using FilterSize = Pow2RoundUp<max(dataSize,filterSize)>;
	using FourierMatrix = Eigen::Matrix<float, FilterSize::value, FilterSize::value>;
	using FourierMatrixPtr = std::shared_ptr<FourierMatrix>;
	using InputMatrixDense = Eigen::Matrix<float, dataSize, dataSize>;
	using InputMatrixSparse = Eigen::SparseMatrix<float>;
	using OutputMatrixDense = Eigen::Matrix<float, dataSize, dataSize>;
	using OutputMatrixSparse = Eigen::SparseMatrix<float>;

	FourierPadder() {};

	/**
	 * @brief Zero-pads a dense input matrix to the next power of 2 (FilterSize::Value)
	 * 
	 * @param  data Dense input matrix
	 * @return Zero-padded dense matrix
	 */
	std::shared_ptr<FourierMatrix> padData(std::shared_ptr<InputMatrixDense> data)
	{
		auto fm = std::make_shared<FourierMatrix>();
		fm->setZero();

		fm->block(0,0,dataSize,dataSize) = *data;

		return fm;
	}

	/**
	 * @brief [brief description]
	 * @details [long description]
	 * 
	 * @param  data Sparse input matrix
	 * @return Zero-padded dense matrix
	 */
	std::shared_ptr<FourierMatrix> padData(std::shared_ptr<InputMatrixSparse> data)
	{
		auto fm = std::make_shared<FourierMatrix>();
		fm->setZero();

		// TODO: Make this faster?! Since the size changes we cant just use the constructor
		// Unless we do a conversativeResize() afterwards
		for (int k=0; k<data->outerSize(); ++k)
		{
			for (InputMatrixSparse::InnerIterator it(*data,k); it; ++it)
			{
				(*fm)(it.row(),it.col()) = it.value();
			}
		}

		return fm;
	}

	/**
	 * @brief Takes a dense matrix of FilterSize::value, and extracts a dense matrix of original data matrix size
	 * 
	 * @param fm Input matrix
	 * @return Dense data matrix
	 */
	std::shared_ptr<OutputMatrixDense> extractDenseOutput(FourierMatrixPtr fm)
	{
		auto dout = std::make_shared<OutputMatrixDense>();
		*dout = fm->block(0,0,dataSize,dataSize);

		return dout;
	}

	/**
	 * @brief Takes a dense matrix of FilterSize::value, and extracts a sparse matrix of original data matrix size
	 * 
	 * @param fm Input matrix
	 * @return Dense data matrix
	 */
	std::shared_ptr<OutputMatrixSparse> extractSparseOutput(FourierMatrixPtr fm)
	{
		auto sout = std::make_shared<OutputMatrixSparse>(dataSize,dataSize);
		*sout = fm->block(0,0,dataSize,dataSize).sparseView();

		return sout;
	}
};

#endif // FOURIER_PADDER_H