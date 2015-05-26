//
// Created by Adam Kosiorek on 26.05.15.
//

#ifndef OPTICAL_FLOW_COMMON_H
#define OPTICAL_FLOW_COMMON_H

#include <memory>
#include <complex>
#include <glog/logging.h>
#include <Eigen/Core>
#include "utils.h"

class Event;
class Filter;
class FilterFactory;
class FourierPadder;
class Quantizer;

using RealMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ComplexMatrix = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#endif //OPTICAL_FLOW_COMMON_H
