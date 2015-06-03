//
// Created by Adam Kosiorek on 26.05.15.
//

#ifndef OPTICAL_FLOW_COMMON_H
#define OPTICAL_FLOW_COMMON_H

#include <memory>
#include <complex>
#include <glog/logging.h>
#include "types.h"
#include "utils.h"

class Filter;
class FilterFactory;
class FourierPadder;

template <template<typename>class BufferType>
class Quantizer;

#endif //OPTICAL_FLOW_COMMON_H
