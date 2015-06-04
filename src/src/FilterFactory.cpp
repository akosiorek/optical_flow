//
// Created by Adam Kosiorek on 14.05.15.
//

#include <complex>
#include <cmath>
#include <utility>

#include "utils.h"
#include "Filter.h"
#include "FilterFactory.h"

//static
const float FilterFactory::PI_ = static_cast<float>(M_PI);

FilterFactory::FilterFactory(float t0, float tk, float tResolution, int xRange, int yRange)
    : xRange_(xRange), yRange_(yRange),
      filterTransformer_([](const MatrixT& filter) { return filter.cast<ComplexMatrix::Scalar>(); }),
    // filters parameters
    sigma(25),
    s1(0.5),
    s2(0.75),
    mu_bi1(0.2),
    sigma_bi1(mu_bi1 / 3),
    mu_bi2(2 * mu_bi1),
    sigma_bi2(1.5 * sigma_bi1),
    mu_mono(0.2 * (1 + mu_bi1 * sqrt(36 + 10 * log(s1 / s2)))),
    sigma_mono(mu_mono / 3),
    fxy(sqrt(2) * 0.08, 0)  {

    LOG_FUN_START;

    if(t0 >= tk) {
        THROW_INVALID_ARG("t0 must be >= tk");
    }
    if(tResolution == 0) {
        THROW_INVALID_ARG("Time Resoluton must be > 0");
    }
    if(xRange_ <= 0 || yRange_ <= 0) {
        THROW_INVALID_ARG("x and y ranges have to be positive");
    }

    timeSpan_ = (tk - t0) / tResolution;
    t0_ = t0 / tResolution;
    xSize_ = 2 * xRange_ + 1;
    ySize_ = 2 * yRange_ + 1;

    LOG_FUN_END;
}

void FilterFactory::setFilterTransformer(FilterTransformT transform) {
    LOG_FUN;
    this->filterTransformer_ = transform;
}

std::shared_ptr <Filter> FilterFactory::createFilter(int angle) const {
    LOG_FUN_START;
    LOG(INFO) << "Creating " << angle << " degree filter";

    auto filters = std::make_unique<std::vector<FilterT>>();
    filters->reserve(timeSpan_);

    float fx, fy; // frequencies;
    std::tie(fx, fy) = rotate(angle, fxy);

    MatrixT spatialRe(xSize_, ySize_);
    MatrixT spatialIm(xSize_, ySize_);
    for(int x = 0; x < xSize_; ++x) {
        for(int y = 0; y < ySize_; ++y) {
            auto v = spatial(x - xRange_, y - yRange_, fx, fy);
            spatialRe(x, y) = v.real();
            spatialIm(x, y) = v.imag();
        }
    }

    float currentTime = t0_ * tResolution_;
    for(int i = 0; i < timeSpan_; ++i) {

        filters->push_back(filterTransformer_(
                spatialIm * timeMono(currentTime) + spatialRe * timeBi(currentTime)));

        currentTime += tResolution_;
    }

    LOG_FUN_END;
    return std::make_shared<Filter>(angle, std::move(filters));
}

std::pair<float, float> FilterFactory::rotate(int angle, const std::pair<float, float> &vec) const {
    LOG_FUN_START;

    float rad = deg2rad(angle);
    float x = std::cos(rad) * vec.first - std::sin(rad) * vec.second;
    float y = std::sin(rad) * vec.first + std::cos(rad) * vec.second;

    LOG_FUN_END;
    return std::make_pair(x, y);
}

float FilterFactory::gaus(float sigma, float mu, float x) const {
    LOG_FUN;
    float tmp = (x - mu) / sigma;
    return std::exp(-0.5 * tmp * tmp);
}

std::complex<float> FilterFactory::spatial(float x, float y, float fx, float fy) const {
    LOG_FUN_START;

    static const std::complex<float> i(0, 1);

    std::complex<float> s = 2 * PI_ / (sigma * sigma);
    s *= std::exp(2 * PI_ * i * (fx*x + fy*y));
    float tmp = PI_ / sigma;
    s *= std::exp(-2 * tmp * tmp * (x*x + y*y));

    LOG_FUN_END;
    return s;
}

float FilterFactory::timeMono(float t) const {
    LOG_FUN;
    return gaus(sigma_mono, mu_mono, t);
}

float FilterFactory::timeBi(float t) const {
    LOG_FUN;
    return -s1 * gaus(sigma_bi1, mu_bi1, t) + s2 * gaus(sigma_bi2, mu_bi2, t);
}
