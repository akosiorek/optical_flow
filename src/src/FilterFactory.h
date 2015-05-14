//
// Created by Adam Kosiorek on 14.05.15.
//

#ifndef OPTICAL_FLOW_FILTERFACTORY_H
#define OPTICAL_FLOW_FILTERFACTORY_H

#include <memory>

class Filter;

class FilterFactory {
public:
    FilterFactory(float t0, float tk, float tResolution, int xRange, int yRange);
    std::shared_ptr<Filter> createFilter(int angle);

private:
    float t0_;
    float tk_;
    float tResolution_;
    int xRange_;
    int yRange_;
};


#endif //OPTICAL_FLOW_FILTERFACTORY_H
