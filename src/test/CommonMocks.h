//
// Created by Adam Kosiorek on 6/13/15.
//

#ifndef OPTICAL_FLOW_COMMONMOCKS_H
#define OPTICAL_FLOW_COMMONMOCKS_H

#include "common.h"
#include "IFilterFactory.h"
#include "IFourierTransformer.h"


struct FilterFactoryMock : public IFilterFactory {

    FilterFactoryMock(int filterSize) : filterSize_(filterSize) {}

    /**
     * create 1x1x1 filter with the only coefficient equal to angle
     */
    virtual std::shared_ptr<Filter> createFilter(int angle) const override;
    virtual void setFilterTransformer(FilterTransformT transform) override;

    int filterSize_;
    FilterTransformT filterTransformer_;
};

struct FourierTransformerMock : public IFourierTransformer {
    virtual void forward(const RealMatrix& src, ComplexMatrix& dst) const override;
    virtual void backward(const ComplexMatrix& src, RealMatrix& dst) const override;
};


#endif //OPTICAL_FLOW_COMMONMOCKS_H
