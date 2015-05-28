#ifndef FOURIER_TRANSFORMER_FFTW_H
#define FOURIER_TRANSFORMER_FFTW_H

#include <complex>

#include <Eigen/Core>

#include <fftw3.h> 

#include "types.h"
#include "IFourierTransformer.h"

class FourierTransformerFFTW : public IFourierTransformer
{
public:

    FourierTransformerFFTW(int rows, int cols);
    ~FourierTransformerFFTW();

    /**
     * @brief Performs forward 2D Fourier transform of real-valued data
     * Takes dst as an argument to allow memory reuse
     *
     * @param src real-valued matrix to be transformed
     * @param dst complex-valued output matrix with the same size as src
     */
    virtual void forward(const RealMatrix& src, ComplexMatrix& dst) const override;

    /**
     * @brief Performs inverse 2D Fourier transform
     * Takes dst as an argument to allow memory reuse
     *
     * @param src complex-valued matrix to be inversly transformed
     * @param dst real-valued output matrix with the same size as src
     */
    virtual void backward(const ComplexMatrix& src, RealMatrix& dst) const override;

    const int rows_;
    const int cols_;
    const int colsHS_;

private:

    inline 
    fftwf_complex* fftw_cast(const std::complex<float> * p) const
    { 
        return const_cast<fftwf_complex*>( reinterpret_cast<const fftwf_complex*>(p) ); 
    }


    fftwf_plan fwd_plan_;
    fftwf_plan bwd_plan_;
};

#endif //FOURIER_TRANSFORMER_FFTW_H
