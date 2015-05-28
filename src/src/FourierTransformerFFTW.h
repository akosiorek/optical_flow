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

    const int rows_;    /*!< Expected rows of real and complex data */
    const int cols_;    /*!< Expected columns of real data only */
    const int colsHS_;  /*!< The number of columns require/present in the complex spectrum */

private:

    /**
     * @brief Casts std::complex<float> pointer to fftwf_complex*
     * @details FFTW expects pointer to complex data to be fftwf_complex, for complex float data.
     *          Additionally, it requires pointers to pointing to non const data. This function will
     *          cast the pointer and removes const'ness.
     * 
     * @param p std::complex<float> data pointer
     * @return fftwf_complex* pointer
     */
    inline 
    fftwf_complex* fftw_cast(const std::complex<float> * p) const
    { 
        return const_cast<fftwf_complex*>( reinterpret_cast<const fftwf_complex*>(p) ); 
    }


    fftwf_plan fwd_plan_;   /*!< Reusable plan for forward transformation */
    fftwf_plan bwd_plan_;   /*!< Reusable plan for backward transformation */
};

#endif //FOURIER_TRANSFORMER_FFTW_H
