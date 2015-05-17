//
// Created by dadrian on 5/14/2015.
//
#ifndef FOURIER_PADDER_H
#define FOURIER_PADDER_H

#include <memory>
#include <stdint.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

class FourierPadder
{
public:
	FourierPadder(const int inputSize, const int filterSize) : 
		inputSize_(inputSize), 
		filterSize_(filterSize)
	{
		fourierSize_ = getNextPowerOfTwo(inputSize_+filterSize_-1);
	}

	std::shared_ptr<Eigen::Matrix<float, 15, 15> > 
	padInput(std::shared_ptr<Eigen::SparseMatrix<float> > input);

	uint32_t getNextPowerOfTwo(uint32_t n) const
	{
		n--;
		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		n |= n >> 8;
		n |= n >> 16;
		return ++n;
	}

private:
	const uint32_t inputSize_;
	const uint32_t filterSize_;
	uint32_t fourierSize_;

};

#endif // FOURIER_PADDER_H