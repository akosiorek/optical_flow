//
// Created by Adam Kosiorek on 6/11/15.
//

#include <glog/logging.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "gpu_math.h"

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

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
const char* cublasGetErrorString(cublasStatus_t error);

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

template <>
void gpu_axpy<float>(const int N, const float alpha, const float* X, float* Y) {
    CUBLAS_CHECK(cublasSaxpy(Cuda::cublasHandle(), N, &alpha, X, 1, Y, 1));
}

template <>
void gpu_axpy<thrust::complex<float> >(const int N, const float alpha,
const thrust::complex<float>* X, thrust::complex<float>* Y) {
    thrust::complex<float> cAlpha(alpha, 0);
    CUBLAS_CHECK(cublasCaxpy(Cuda::cublasHandle(), N, (cuComplex*)&cAlpha, (cuComplex*)X, 1, (cuComplex*)Y, 1));
}

template <>
void gpu_scal<float>(const int N, const float alpha, float *X) {
    CUBLAS_CHECK(cublasSscal(Cuda::cublasHandle(), N, &alpha, X, 1));
}

template <>
void gpu_scal<thrust::complex<float> >(const int N, const float alpha, thrust::complex<float> *X) {
    thrust::complex<float> cAlpha(alpha, 0);
    CUBLAS_CHECK(cublasCscal(Cuda::cublasHandle(), N, (cuComplex*)&alpha, (cuComplex*)X, 1));
}

template <class Dtype>
__global__ void add_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {

    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] + b[index];
    }
}

template <>
void gpu_add<float>(const int N, const float* a, const float* b, float* y) {

    add_kernel<float><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <>
void gpu_add<thrust::complex<float> >(const int N, const thrust::complex<float>* a,
    const thrust::complex<float>* b, thrust::complex<float>* y) {

    add_kernel<thrust::complex<float> ><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
}

template <class Dtype>
__global__ void mul_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {

    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] * b[index];
    }
}

template <>
void gpu_mul<float >(const int N, const float* a, const float* b, float* y) {

    mul_kernel <float><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>> (N, a, b, y);
}

template <>
void gpu_mul<thrust::complex<float> >(const int N, const thrust::complex<float>* a,
    const thrust::complex<float>* b, thrust::complex<float>* y) {

    mul_kernel<thrust::complex<float> ><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>> (N, a, b, y);
}
