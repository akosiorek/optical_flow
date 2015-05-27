//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_IFILTERFACTORY_H
#define OPTICAL_FLOW_IFILTERFACTORY_H

#include "common.h"
#include "Filter.h"

class IFilterFactory {
public:
    using MatrixT = Filter::MatrixT;
    using FilterT = Filter::FilterT;
    using FilterTransformT = std::function<FilterT(const MatrixT&)>;

public:
    virtual ~IFilterFactory() = default;
    virtual void setFilterTransformer(FilterTransformT transform) = 0;
    virtual std::shared_ptr<Filter> createFilter(int angle) const = 0;
};

#endif //OPTICAL_FLOW_IFILTERFACTORY_H
