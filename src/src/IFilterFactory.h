//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_IFILTERFACTORY_H
#define OPTICAL_FLOW_IFILTERFACTORY_H

#include <memory>
#include <complex>
#include <Eigen/Core>

#include "Filter.h"

class IFilterFactory {
public:
    using FilterT = Filter::FilterT;
    using FilterTransformT = std::function<FilterT(const FilterT&)>;

public:
    virtual ~IFilterFactory() = default;
    virtual void setFilterTransformer(FilterTransformT transform) = 0;
    virtual std::shared_ptr<Filter> createFilter(int angle) const = 0;
};

#endif //OPTICAL_FLOW_IFILTERFACTORY_H
