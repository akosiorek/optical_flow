//
// Created by Adam Kosiorek on 23.05.15.
//

#include "IFourierTransformer.h"

Eigen::MatrixXcf IFourierTransformer::forward(const Eigen::MatrixXf &src) const {

    Eigen::MatrixXcf dst(src.rows(), src.cols());
    forward(src, dst);
    return dst;
}

Eigen::MatrixXf IFourierTransformer::backward(const Eigen::MatrixXcf &src) const {

    Eigen::MatrixXf dst(src.rows(), src.cols());
    backward(src, dst);
    return dst;
}