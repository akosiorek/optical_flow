//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_IFILTERFACTORY_H
#define OPTICAL_FLOW_IFILTERFACTORY_H

#include <memory>
#include <complex>
#include <Eigen/Core>

class Filter;

class IFilterFactory {
public:
    virtual ~IFilterFactory() = default;
    virtual std::shared_ptr<Filter> createFilter(int angle) const = 0;
};

#endif //OPTICAL_FLOW_IFILTERFACTORY_H
