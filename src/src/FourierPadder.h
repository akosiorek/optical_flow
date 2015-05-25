#ifndef FOURIER_PADDER_H
#define FOURIER_PADDER_H

#include "common.h"
#include <stdint.h>

#include <Eigen/SparseCore>

// // Determines the next power of two at run time
// template<uint32_t A, uint8_t B = 16>
// struct Pow2RoundUp { enum{ value = Pow2RoundUp<((B == 16 ? (A-1) : A) | ((B == 16 ? (A-1) : A) >> B)), B/2>::value }; };
// template<uint32_t A >
// struct Pow2RoundUp<A, 1> { enum{ value = ((A | (A >> 1)) + 1) }; };

// /**
//  * @brief Constexpr function to determine the max of two values at compile time
//  * 
//  * @param a First value
//  * @param b Second value
//  * 
//  * @return Returns the larger element by value
//  */
// template<typename T> constexpr
// T const& max(T const& a, T const& b)
// {
// 	return a > b ? a : b;
// }

/**
 * @brief This class zero-pads matrices for efficient FFT.
 * @details Matrix size typically needs to be a power of two to be efficiently FFT'ed.
 * 			(Actually, e.g., prime numbers etc. might do it as well, but this is really just for now.)
 * 
 * @tparam int dataSize Size of data matrix
 * @tparam int filterSize Size of filter
 */
class FourierPadder
{
public:
	using Ptr = std::shared_ptr<FourierPadder>;

	using EBOFMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using EBOFMatrixSparse = Eigen::SparseMatrix<float, Eigen::RowMajor>;

	FourierPadder(unsigned int dataSize, unsigned int filterSize) 
		: 	dataSize_(dataSize), 
			filterSize_(filterSize),
			fourierSize_(dataSize+filterSize-1),
			fourierSizePadded_(roundUpPow2(dataSize+filterSize-1)) //linear conv+zeropadding
	{
	}
-
	/**
	 * @brief Zero-pads a dense input matrix to the next power of 2
	 * 
	 * @param  data Dense input matrix
	 * @return Zero-padded dense matrix
	 */
	void padData(const EBOFMatrix& tm, EBOFMatrix& fm)
	{
		if(fm.cols()!=fourierSizePadded_ || fm.rows()!=fourierSizePadded_)
		{
			fm.resize(fourierSizePadded_, fourierSizePadded_);
		}

		fm.setZero();

		fm.block(0,0,dataSize_,dataSize_) = tm; // could use template later on
	}

	/**
	 * @brief Zero-pads a sparse input matrix to the next power of 2
	 * @details [long description]
	 * 
	 * @param  data Sparse input matrix
	 * @return Zero-padded dense matrix
	 */
	void padData(const EBOFMatrixSparse& tm, EBOFMatrix& fm)
	{
		if(fm.cols()!=fourierSizePadded_ || fm.rows()!=fourierSizePadded_)
		{
			fm.resize(fourierSizePadded_, fourierSizePadded_);
		}

		fm.setZero();

		for (int k=0; k<tm.outerSize(); ++k)
		{
			for (EBOFMatrixSparse::InnerIterator it(tm,k); it; ++it)
			{
				fm(it.row(),it.col()) = it.value();
			}
		}
	}

		/**
	 * @brief Zero-pads a dense input matrix to the next power of 2
	 * 
	 * @param  data Dense input matrix
	 * @return Zero-padded dense matrix
	 */
	void padFilter(const EBOFMatrix& tm, EBOFMatrix& fm)
	{
		if(fm.cols()!=fourierSizePadded_ || fm.rows()!=fourierSizePadded_)
		{
			fm.resize(fourierSizePadded_, fourierSizePadded_);
		}

		fm.setZero();

		fm.block(0,0,filterSize_,filterSize_) = tm;
	}

	// TODO: Is this one needed, the dense method also works for sparse matrices. Speed difference?!?!?
	/**
	 * @brief Zero-pads a sparse input matrix to the next power of 2
	 * @details [long description]
	 * 
	 * @param  data Sparse input matrix
	 * @return Zero-padded dense matrix
	 */
	void padFilter(const EBOFMatrixSparse& tm, EBOFMatrix& fm)
	{
		if(fm.cols()!=fourierSizePadded_ || fm.rows()!=fourierSizePadded_)
		{
			fm.resize(fourierSizePadded_, fourierSizePadded_);
		}

		fm.setZero();

		// TODO: Make this faster?! Since the size changes we cant just use the constructor
		// Unless we do a conversativeResize() afterwards
		for (int k=0; k<tm.outerSize(); ++k)
		{
			for (EBOFMatrixSparse::InnerIterator it(tm,k); it; ++it)
			{
				fm(it.row(),it.col()) = it.value();
			}
		}
	}

	/**
	 * @brief Takes a dense matrix of FilterSize::value, and extracts a dense matrix of original data matrix size
	 * 
	 * @param fm Input matrix
	 * @return Dense data matrix
	 */
	void extractDenseOutput(const EBOFMatrix& fm, EBOFMatrix& tm)
	{
		if(tm.cols()!=fourierSize_ || tm.rows()!=fourierSize_)
		{
			tm.resize(fourierSize_, fourierSize_);
		}

		tm.setZero();

		tm = fm.block(0,0,fourierSize_,fourierSize_);
	}

	/**
	 * @brief Takes a dense matrix of FilterSize::value, and extracts a sparse matrix of original data matrix size
	 * 
	 * @param fm Input matrix
	 * @return Dense data matrix
	 */
	void extractSparseOutput(const EBOFMatrix& fm, EBOFMatrixSparse& tm)
	{
		if(tm.cols()!=fourierSize_ || tm.rows()!=fourierSize_)
		{
			tm.resize(fourierSize_, fourierSize_);
		}

		tm.setZero();

		tm = fm.block(0,0,fourierSize_, fourierSize_).sparseView();
	}

	/**
 * @brief Rounds a 32 unsigned int to the closest power of two
 *
 * @param v Input number
 * @return Power of two
 */
	static uint32_t roundUpPow2(uint32_t v)
	{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		return ++v;
	}


	// can be public as they are const
	const unsigned int dataSize_;
	const unsigned int filterSize_;
	const unsigned int fourierSize_;
	const unsigned int fourierSizePadded_;
};

#endif // FOURIER_PADDER_H