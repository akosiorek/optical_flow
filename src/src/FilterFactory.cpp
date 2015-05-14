//
// Created by Adam Kosiorek on 14.05.15.
//

#include "utils.h"
#include "Filter.h"
#include "FilterFactory.h"

FilterFactory::FilterFactory(float t0, float tk, float tResolution, int xRange, int yRange)
    : t0_(t0), tk_(tk), tResolution_(tResolution), xRange_(xRange), yRange_(yRange) {}

std::shared_ptr <Filter> FilterFactory::createFilter(int angle) {
    auto filters = make_unique<std::vector<Eigen::MatrixXf>>();

    //populate filters;


    return std::make_shared<Filter>(angle, std::move(filters));
}
