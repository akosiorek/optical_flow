//
// Created by dadrian on 5/14/2015.
//
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

template<typename T> constexpr
T const& max(T const& a, T const& b)
{
	return a > b ? a : b;
}

// TODO evaluate if we really need to pad up to power of two
template<unsigned int dataSize, unsigned int filterSize>
class FourierPadder
{
public:
	using Ptr = std::shared_ptr<FourierPadder>;

	using FilterSize = Pow2RoundUp<max(dataSize,filterSize)>;
	using FourierMatrix = Eigen::Matrix<float, FilterSize::value, FilterSize::value>;
	using FourierMatrixPtr = std::shared_ptr<FourierMatrix>;
	using InputMatrix = Eigen::SparseMatrix<float>;
	using InputMatrixPtr = std::shared_ptr<InputMatrix>;
	using OutputMatrixDense = Eigen::Matrix<float, dataSize, dataSize>;
	using OutputMatrixSparse = Eigen::SparseMatrix<float>;

	FourierPadder() {};

	std::shared_ptr<FourierMatrix> padData(std::shared_ptr<InputMatrix> data)
	{
		auto fm = std::make_shared<FourierMatrix>();
		fm->setZero();

		// TODO: Make this faster?! Since the size changes we cant just use the constructor
		// Unless we do a conversativeResize() afterwards
		for (int k=0; k<data->outerSize(); ++k)
		{
			for (InputMatrix::InnerIterator it(*data,k); it; ++it)
			{
				(*fm)(it.row(),it.col()) = it.value();
			}
		}

		return fm;
	}

	std::shared_ptr<OutputMatrixDense> extractDenseOutput(FourierMatrixPtr fm)
	{
		auto dout = std::make_shared<OutputMatrixDense>();
		*dout = fm->block(0,0,dataSize,dataSize);

		return dout;
	}

	std::shared_ptr<OutputMatrixSparse> extractSparseOutput(FourierMatrixPtr fm)
	{
		auto sout = std::make_shared<OutputMatrixSparse>(dataSize,dataSize);
		*sout = fm->block(0,0,dataSize,dataSize).sparseView();

		return sout;
	}

private:


};

#endif // FOURIER_PADDER_H