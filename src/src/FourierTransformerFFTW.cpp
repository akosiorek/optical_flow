#include <Eigen/Core>

#include "utils.h"
#include "FourierTransformerFFTW.h"

FourierTransformerFFTW::FourierTransformerFFTW(int rows, int cols)
	: 	rows_(rows),
		cols_(cols),
		colsHS_(cols/2 + 1)
{
	// Allocate some bogus space // Needed for FFTW_MEASURE
	float* rl = new float[rows_*cols_];
	// In case of real2complex, we only need allocate space for the half-spectrum
	std::complex<float>* cl = new std::complex<float>[rows_ * colsHS_];

	/* Create plan */
	//TODO NEEDS THREAD SAFETY IN THE FUTURE!?
	fwd_plan_ = fftwf_plan_dft_r2c_2d(rows_, cols_, rl, fftw_cast(cl), FFTW_MEASURE);
	bwd_plan_ = fftwf_plan_dft_c2r_2d(rows_, cols_, fftw_cast(cl), rl,FFTW_MEASURE);
	delete[] rl;
	delete[] cl;
}

FourierTransformerFFTW::~FourierTransformerFFTW()
{
	/* Free plan memory */
	fftwf_destroy_plan(fwd_plan_); //TODO: possible thread safety...
	fftwf_destroy_plan(bwd_plan_); //TODO: possible thread safety...
}

void FourierTransformerFFTW::forward(const RealMatrix& src, ComplexMatrix& dst) const 
{
	// TODO: FFTW can also accept a reverse order of dimensions? how does this effect the result?
	// (Instead of row-major ordering...but right now we can change that when padding anyway)

	// check if row-major storage order is present (as required by fftw)
	static_assert(src.IsRowMajor,"Input: Eigen Matrix is not given in row-major storage order!");
	static_assert(dst.IsRowMajor,"Output: Eigen Matrix is not given in row-major storage order!");

	// check if sizes are matching the sizes given in the constructor
	if(src.rows()!=rows_ || src.cols()!=cols_)
	{
		THROW_INVALID_ARG("Size of input matrix did not match the configuration");
	}
	if(dst.rows()!=rows_ || dst.cols()!=colsHS_)
	{
		THROW_INVALID_ARG("Size of output matrix did not match the configuration");
	}

	/* Compute forward DFT */
	fftwf_execute_dft_r2c(fwd_plan_, const_cast<float*>(src.data()), fftw_cast(dst.data()));
}

// DANGER: THIS FUCKS UP THE INPUT ARRAY SRC (OR USE PRESERVER FLAG)
void FourierTransformerFFTW::backward(const ComplexMatrix& src, RealMatrix& dst) const
{
	// TODO: FFTW can also accept a reverse order of dimensions? how does this effect the result?

	// check if row-major storage order is present (as required by fftw)
	static_assert(src.IsRowMajor,"Input: Eigen Matrix is not given in row-major storage order!");
	static_assert(dst.IsRowMajor,"Output: Eigen Matrix is not given in row-major storage order!");

	// check if sizes are matching the sizes given in the constructor
	if(src.rows()!=rows_ || src.cols()!=colsHS_)
	{
		THROW_INVALID_ARG("Size of input matrix did not match the configuration");
	}
	if(dst.rows()!=rows_ || dst.cols()!=cols_)
	{
		THROW_INVALID_ARG("Size of output matrix did not match the configuration");		
	}

	/* Compute forward DFT */
	fftwf_execute_dft_c2r(bwd_plan_, fftw_cast(src.data()), dst.data());
}
