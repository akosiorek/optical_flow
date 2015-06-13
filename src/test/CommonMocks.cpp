//
// Created by Adam Kosiorek on 6/13/15.
//

#include "CommonMocks.h"

//  ### FilterFactoryMock   ###################################################
std::shared_ptr<Filter> FilterFactoryMock::createFilter(int angle) const {

    auto filters = std::make_unique<std::vector<FilterT>>();

    MatrixT mat(filterSize_, filterSize_);
    mat.setConstant(angle);
    filters->emplace_back(filterTransformer_(mat));
    return std::make_shared<Filter>(angle, std::move(filters));
}

void FilterFactoryMock::setFilterTransformer(FilterTransformT transform) {
    filterTransformer_ = transform;
}


//  ### FourierTransformerMock   ###################################################
void FourierTransformerMock::forward(const RealMatrix& src, ComplexMatrix& dst) const {

    dst = src.cast<ComplexMatrix::Scalar>();
}

void FourierTransformerMock::backward(const ComplexMatrix& src, RealMatrix& dst) const {
    dst = src.real();
}