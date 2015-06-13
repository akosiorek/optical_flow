//
// Created by Adam Kosiorek on 6/11/15.
//

#include <glog/logging.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "GpuMath.h"
#include "CudaUtils.h"

//  ###########################################################################
//  ### Helpers  ##############################################################
//  ###########################################################################

// CUDA: various checks for different function calls.

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#if __CUDA_ARCH__ >= 200
    const int CUDA_NUM_THREADS = 1024;
#else
    const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}


class Cuda {
public:
    static Cuda& get() {
        static Cuda cuda_;
        return cuda_;
    }

    static cublasHandle_t cublasHandle() {
        return get().cublasHandle_;
    }
private:
    Cuda() {
        if (cublasCreate(&cublasHandle_) != CUBLAS_STATUS_SUCCESS) {
            LOG(FATAL) << "Cannot create Cublas handle";
        }
    }

    ~Cuda() {
        if (cublasHandle_) {
            CUBLAS_CHECK(cublasDestroy(cublasHandle_));
        }
    }

private:
    cublasHandle_t cublasHandle_;
};



//  ###########################################################################
//  ### Math Functions  #######################################################
//  ###########################################################################



template <>
void gpuAXPY<float>(const int N, const float alpha, const float* X, float* Y) {
    CUBLAS_CHECK(cublasSaxpy(Cuda::cublasHandle(), N, &alpha, X, 1, Y, 1));
}

template <>
void gpuAXPY<thrust::complex<float> >(const int N, const float alpha,
const thrust::complex<float>* X, thrust::complex<float>* Y) {
    thrust::complex<float> cAlpha(alpha, 0);
    CUBLAS_CHECK(cublasCaxpy(Cuda::cublasHandle(), N, (cuComplex*)&cAlpha, (cuComplex*)X, 1, (cuComplex*)Y, 1));
}

template <>
void gpuScale<float>(const int N, const float alpha, float *X) {
    CUBLAS_CHECK(cublasSscal(Cuda::cublasHandle(), N, &alpha, X, 1));
}

template <>
void gpuScale<thrust::complex<float> >(const int N, const float alpha, thrust::complex<float> *X) {
    thrust::complex<float> cAlpha(alpha, 0);
    CUBLAS_CHECK(cublasCscal(Cuda::cublasHandle(), N, (cuComplex*)&alpha, (cuComplex*)X, 1));
}

template <class Dtype>
__global__ void mulToKernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {

    CUDA_KERNEL_LOOP(index, n) {
        y[index] += a[index] * b[index];
    }
}

template <>
void gpuMulTo<float >(const int N, const float* a, const float* b, float* y) {

    mulToKernel<float><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>> (N, a, b, y);
}

template <>
void gpuMulTo<thrust::complex<float> >(const int N, const thrust::complex<float>* a,
    const thrust::complex<float>* b, thrust::complex<float>* y) {

    mulToKernel<thrust::complex<float> ><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>> (N, a, b, y);
}
