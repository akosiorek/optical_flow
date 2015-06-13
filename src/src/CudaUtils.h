//
// Created by Adam Kosiorek on 6/13/15.
//

#ifndef OPTICAL_FLOW_CUDAUTILS_H
#define OPTICAL_FLOW_CUDAUTILS_H

#include <glog/logging.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#endif //OPTICAL_FLOW_CUDAUTILS_H
