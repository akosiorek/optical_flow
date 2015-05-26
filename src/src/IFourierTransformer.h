//
// Created by Adam Kosiorek on 23.05.15.
//

#ifndef OPTICAL_FLOW_IFOURIERTRANSFORMER_H
#define OPTICAL_FLOW_IFOURIERTRANSFORMER_H

#include <Eigen/Core>

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
    virtual void forward(const Eigen::MatrixXf& src, Eigen::MatrixXcf& dst) const = 0;

    /**
     * @brief Performs inverse 2D Fourier transform
     * Takes dst as an argument to allow memory reuse
     *
     * @param src complex-valued matrix to be inversly transformed
     * @param dst real-valued output matrix with the same size as src
     */
    virtual void backward(const Eigen::MatrixXcf& src, Eigen::MatrixXf& dst) const = 0;

    /**
     * @brief Performs forward 2D Fourier transform
     *
     * @param src real-valued matrix to be transformed
     * @return complex-valued output matrix with the same size as src
     */
    Eigen::MatrixXcf forward(const Eigen::MatrixXf& src) const;

    /**
     * @brief Performs inverse 2D Fourier transform
     *
     * @param src complex-valued matrix to be inversly transformed
     * @return real-valued output matrix with the same size as src
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXcf& src) const;
};

#endif //OPTICAL_FLOW_IFOURIERTRANSFORMER_H
