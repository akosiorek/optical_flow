//
// Created by Adam Kosiorek on 14.05.15.
//

#ifndef OPTICAL_FLOW_FILTERFACTORY_H
#define OPTICAL_FLOW_FILTERFACTORY_H

#include "common.h"
#include "IFilterFactory.h"

class FilterFactory : public IFilterFactory {
public:
    typedef std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> FilterTransformT;
public:
    FilterFactory(float t0, float tk, float tResolution, int xRange, int yRange);
    void setFilterTransformer(FilterTransformT transform);
    std::shared_ptr<Filter> createFilter(int angle) const override;

private:
    std::pair<float, float> rotate(int angle, const std::pair<float, float>& vec) const;
    float gaus(float sigma, float mu, float x) const;
    std::complex<float> spatial(float x, float y, float fx, float fy) const;
    float timeMono(float t) const;
    float timeBi(float t) const;

private:
    int t0_;
    int timeSpan_;
    float tResolution_;
    int xRange_;
    int yRange_;
    int xSize_;
    int ySize_;

private:
    FilterTransformT filterTransformer_;
 
    // filter parameters from paper
    float sigma;
    float s1;
    float s2;
    float mu_bi1;
    float sigma_bi1;
    float mu_bi2;
    float sigma_bi2;
    float mu_mono;
    float sigma_mono;
    std::pair<float, float> fxy;

    static const float PI_;
};


#endif //OPTICAL_FLOW_FILTERFACTORY_H
