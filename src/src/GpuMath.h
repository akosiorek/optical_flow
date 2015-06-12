//
// Created by Adam Kosiorek on 6/11/15.
// Code mostly copied from https://github.com/BVLC/caffe
//

#ifndef OPTICAL_FLOW_GPU_MATH_H
#define OPTICAL_FLOW_GPU_MATH_H

/**
 * Y = alpha * X + Y
 */
template <class Dtype>
void gpuAXPY(const int N, const float alpha, const Dtype* X, Dtype* Y);

/**
 * X = alpha * X
 */
template <class Dtype>
void gpuScale(const int N, const float alpha, Dtype *X);

/**
 * y = y + a * b
 */
template <class Dtype>
void gpuMulTo(const int N, const Dtype* a, const Dtype* b, Dtype* y);

#endif //OPTICAL_FLOW_GPU_MATH_H
