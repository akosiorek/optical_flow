//
// Created by Adam Kosiorek on 23.05.15.
//

#ifndef OPTICAL_FLOW_IFOURIERTRANSFORMER_H
#define OPTICAL_FLOW_IFOURIERTRANSFORMER_H

#include <Eigen/Core>

#include "types.h"

/**
 * @brief An interface for Fourier Transform utility
 */
class IFourierTransformer {
public:

    /**
     * @brief Performs forward 2D Fourier transform
     * Takes dst as an argument to allow memory reuse
     *
     * @param src real-valued matrix to be transformed
     * @param dst complex-valued output matrix with the same size as src
     */
    virtual void forward(const RealMatrix& src, ComplexMatrix& dst) const = 0;

    /**
     * @brief Performs inverse 2D Fourier transform
     * Takes dst as an argument to allow memory reuse
     *
     * @param src complex-valued matrix to be inversly transformed
     * @param dst real-valued output matrix with the same size as src
     */
    virtual void backward(const ComplexMatrix& src, RealMatrix& dst) const = 0;

    /**
     * @brief Performs forward 2D Fourier transform
     *
     * @param src real-valued matrix to be transformed
     * @return complex-valued output matrix with the same size as src
     */
    ComplexMatrix forward(const RealMatrix& src) const
    {
        ComplexMatrix dst(src.rows(), src.cols());
        forward(src, dst);
        return dst;
    }

    /**
     * @brief Performs inverse 2D Fourier transform
     *
     * @param src complex-valued matrix to be inversly transformed
     * @return real-valued output matrix with the same size as src
     */
    RealMatrix backward(const ComplexMatrix& src) const
    {
        RealMatrix dst(src.rows(), src.cols());
        backward(src, dst);
        return dst;
    }
};

#endif //OPTICAL_FLOW_IFOURIERTRANSFORMER_H
