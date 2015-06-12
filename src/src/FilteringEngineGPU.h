//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_FILTERINGENGINECPU_H
#define OPTICAL_FLOW_FILTERINGENGINECPU_H

#include <cuda.h>
#include <thrust/complex.h>
#include <boost/circular_buffer.hpp>
#include "FilteringEngine.h"
#include "DeviceBlob.h"
#include "gpu_math.h"

thrust::complex<float>* castThrust(std::complex<float>* ptr) {
    return reinterpret_cast<thrust::complex<float>*>(ptr);
}

template<template<class> class InputBufferT, template<class> class OutputBufferT = InputBufferT>
class FilteringEngineGPU : public FilteringEngine<InputBufferT, OutputBufferT> {
public:
    using BaseT = FilteringEngine<InputBufferT, OutputBufferT>;
    using BaseT::timeSteps_;
private:
    using RealBlob = DeviceBlob<float>;
    using ComplexBlob = DeviceBlob<thrust::complex<float>>;
public:

    FilteringEngineGPU(std::unique_ptr<IFilterFactory> factory,
                    std::unique_ptr<FourierPadder> padder,
                    std::unique_ptr<IFourierTransformer> transformer)

            : BaseT(std::move(factory), std::move(padder), std::move(transformer)),
            eventBuffer_(0),
            rowsTransformed_(0),
            colsTransformed_(0) {}


    virtual void storeFilter(std::shared_ptr<Filter> filter) override {

        this->angles_.push_back(filter->angle());
        this->filters_.emplace_back();
        auto& filterVec = this->filters_[this->filters_.size() - 1];
        filterVec.reserve(filter->numSlices());

        ComplexMatrix transformed;
        for(int i = 0; i < filter->numSlices(); ++i) {
            this->transformer_->forward(filter->at(i), transformed);
            filterVec.emplace_back(transformed.rows(), transformed.cols(), castThrust(transformed.data()));
        }
        if(rowsTransformed_ == 0) {
            rowsTransformed_ = transformed.rows();
            colsTransformed_ = transformed.cols();
        }
    }

    virtual void prepareResponseBuffer() override {
        responseBuffer_.emplace_back(rowsTransformed_, colsTransformed_);
        if(finalResponseBufferX_.rows() != rowsTransformed_) {
            finalResponseBufferX_ = ComplexBlob(rowsTransformed_, colsTransformed_);
            finalResponseBufferY_ = ComplexBlob(rowsTransformed_, colsTransformed_);
        }
    }

    virtual bool isInitialized() override {
        return  !this->angles_.empty() && eventBuffer_.size() == timeSteps_;
    }

    virtual void initialize(std::shared_ptr<Filter> filter) override {
        // intialize buffer by allocating memory for all event slices to be kept
        if(eventBuffer_.size() != timeSteps_) {
            eventBuffer_.set_capacity(timeSteps_);

            while(eventBuffer_.size() != eventBuffer_.capacity()) {
                eventBuffer_.push_back(ComplexBlob(rowsTransformed_, colsTransformed_));
            }
            extractedDataBuffer_.resize(filter->xSize(), filter->ySize());
        }
    }

    virtual void filter(std::shared_ptr<EventSlice> slice) override {
        LOG_FUN_START;

        eventBuffer_.rotate(eventBuffer_.end() - 1);
        ++this->receivedEventSlices_;
        this->padder_->padData(*slice, paddedDataBuffer_);
        this->transformer_->forward(paddedDataBuffer_, transformedDataBuffer_);
        eventBuffer_[0].copyFrom(castThrust(transformedDataBuffer_.data()));


        // may the magic happen
        if(this->isBufferFilled()) {

            // zero the response buffers
            for(auto& buffer : responseBuffer_) {
                buffer.setZero();
            }

            // iterate over eventSlices and filterSlices
            for(int sliceIndex = 0; sliceIndex < timeSteps_; ++sliceIndex) {
                const auto& eventSlice = eventBuffer_[sliceIndex];
                // iterate over filters
                for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {
                    const auto& filterSlice = filters_[filterIndex][sliceIndex];
//                    responseBuffer_[filterIndex] += eventSlice * filterSlice;
                    gpu_mul(eventSlice.count(), eventSlice.data(), filterSlice.data(), responseBuffer_[filterIndex].data());
                }
            }

            auto flowSlice = std::make_shared<FlowSlice>(slice->rows(), slice->cols());
            finalResponseBufferX_.setZero();
            finalResponseBufferY_.setZero();
            for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {

                float rad = deg2rad(this->angles_[filterIndex]);
                gpu_axpy(finalResponseBufferX_.count(), std::cos(rad), responseBuffer_[filterIndex].data(), finalResponseBufferX_.data());
                gpu_axpy(finalResponseBufferY_.count(), -std::sin(rad), responseBuffer_[filterIndex].data(), finalResponseBufferY_.data());
            }

            finalResponseBufferX_.copyTo(castThrust(transformedDataBuffer_.data()));
            this->transformer_->backward(transformedDataBuffer_, inversedDataBuffer_);
            this->padder_->extractDenseOutput(inversedDataBuffer_, flowSlice->xv_);

            finalResponseBufferY_.copyTo(castThrust(transformedDataBuffer_.data()));
            this->transformer_->backward(transformedDataBuffer_, inversedDataBuffer_);
            this->padder_->extractDenseOutput(inversedDataBuffer_, flowSlice->yv_);

            this->outputBuffer_->push(flowSlice);
        }

        LOG_FUN_END;
    }

    /**
     * Returns number of filters;
     */
    virtual int numFilters() override {
        LOG_FUN;
        return filters_.size();
    }

private:
    RealMatrix paddedDataBuffer_;
    RealMatrix extractedDataBuffer_;
    RealMatrix inversedDataBuffer_;
    ComplexMatrix transformedDataBuffer_;
    ComplexBlob finalResponseBufferX_;
    ComplexBlob finalResponseBufferY_;
    std::vector<std::vector<ComplexBlob>> filters_;
    std::vector<ComplexBlob> responseBuffer_;
    boost::circular_buffer<ComplexBlob> eventBuffer_;
    int rowsTransformed_;
    int colsTransformed_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINECPU_H
