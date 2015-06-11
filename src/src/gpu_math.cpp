//
// Created by Adam Kosiorek on 6/11/15.
//

#include <complex>
#include <thrust/complex.h>
#include "gpu_math.h"

using ComplexGPU = thrust::complex<float>;
using ComplexCPU = std::complex<float>;

template <>
void gpu_axpy<ComplexCPU>(const int N, const float alpha, const ComplexCPU* X, ComplexCPU* Y) {
    gpu_axpy(N, alpha, (ComplexGPU*)X, (ComplexGPU*)Y);
}

template <>
void gpu_scal<ComplexCPU>(const int N, const float alpha, ComplexCPU *X) {
    gpu_scal(N, alpha, (ComplexGPU*)X);
}


template <>
void gpu_add<ComplexCPU>(const int N, const ComplexCPU* a, const ComplexCPU* b, ComplexCPU* y) {
    gpu_add(N, (ComplexGPU*)a, (ComplexGPU*)b, (ComplexGPU*)y);
}


template <>
void gpu_mul<ComplexCPU>(const int N, const ComplexCPU* a, const ComplexCPU* b, ComplexCPU* y) {
    gpu_mul(N, (ComplexGPU*)a, (ComplexGPU*)b, (ComplexGPU*)y);
}


