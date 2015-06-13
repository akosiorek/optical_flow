//
// Created by Adam Kosiorek on 6/12/15.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "DeviceBlob.h"
#include "CudaUtils.h"

template<class Dtype>
DeviceBlob<Dtype>::DeviceBlob() : rows_(0), cols_(0), count_(0), bytes_(0), data_(NULL) {}

template<class Dtype>
DeviceBlob<Dtype>::DeviceBlob(int rows, int cols)
        : rows_(rows), cols_(cols), count_(rows_ * cols_), bytes_(count_ * sizeof(Dtype)) {
    CUDA_CHECK(cudaMalloc(&data_, bytes_));
}

template<class Dtype>
DeviceBlob<Dtype>::DeviceBlob(int rows, int cols, const Dtype* from) : DeviceBlob(rows, cols) {
    copyFrom(from);
}

template<class Dtype>
DeviceBlob<Dtype>::DeviceBlob(const DeviceBlob& that) : DeviceBlob(that.rows(), that.cols()) {
    if(that.data() != NULL && bytes_ != 0) {
        CUDA_CHECK(cudaMemcpy(data_, that.data(), bytes_, cudaMemcpyDeviceToDevice));
    }
}

template<class Dtype>
DeviceBlob<Dtype>::~DeviceBlob() {
    cudaFree(data_);
    data_ = NULL;
}

template<class Dtype>
DeviceBlob<Dtype>& DeviceBlob<Dtype>::operator= (DeviceBlob that) {
    swap(*this, that);
    return *this;
}


template<class Dtype>
void DeviceBlob<Dtype>::copyFrom(const Dtype* from) {
    CUDA_CHECK(cudaMemcpy(data_, from, bytes_, cudaMemcpyHostToDevice));
}

template<class Dtype>
void DeviceBlob<Dtype>::copyTo(Dtype* to) const {
    CUDA_CHECK(cudaMemcpy(to, data_, bytes_, cudaMemcpyDeviceToHost));
}

template<class Dtype>
void DeviceBlob<Dtype>::setZero() {
    CUDA_CHECK(cudaMemset(data_, 0, bytes_));
}

template<class Dtype>
size_t DeviceBlob<Dtype>::rows() const {
    return rows_;
}

template<class Dtype>
size_t DeviceBlob<Dtype>::cols() const {
    return cols_;
}

template<class Dtype>
size_t DeviceBlob<Dtype>::count() const {
    return count_;
}

template<class Dtype>
Dtype* DeviceBlob<Dtype>::data() {
    return data_;
}

template<class Dtype>
const Dtype* DeviceBlob<Dtype>::data() const{
    return data_;
}

template class DeviceBlob<float>;
template class DeviceBlob<thrust::complex<float> >;