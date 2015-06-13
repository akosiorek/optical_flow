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
#include "GpuMath.h"

#define LOG_DIMS(x) LOG(ERROR) << #x << " rows: " << x.rows() << " cols: " << x.cols()


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
            colsTransformed_(0) { LOG_FUN; }


    virtual void storeFilter(std::shared_ptr<Filter> filter) override {
        LOG_FUN_START;

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

        LOG_FUN_END;
    }

    virtual void prepareResponseBuffer() override {
        LOG_FUN_START;

        responseBuffer_.emplace_back(rowsTransformed_, colsTransformed_);
        if(weightedResponseBufferX_.rows() != rowsTransformed_) {
            weightedResponseBufferX_ = ComplexBlob(rowsTransformed_, colsTransformed_);
            weightedResponseBufferY_ = ComplexBlob(rowsTransformed_, colsTransformed_);
        }

        LOG_FUN_END;
    }

    virtual bool isInitialized() override {
        LOG_FUN;
        return  !this->angles_.empty() && eventBuffer_.size() == timeSteps_;
    }

    virtual void initialize(std::shared_ptr<Filter> filter) override {
        LOG_FUN_START;

        // intialize buffer by allocating memory for all event slices to be kept
        if(eventBuffer_.size() != timeSteps_) {
            eventBuffer_.set_capacity(timeSteps_);

            while(eventBuffer_.size() != eventBuffer_.capacity()) {
                eventBuffer_.push_back(ComplexBlob(rowsTransformed_, colsTransformed_));
            }
        }

        LOG_FUN_END;
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
                #pragma omp parallel for
                for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {
                    const auto& filterSlice = filters_[filterIndex][sliceIndex];
                    gpuMulTo(eventSlice.count(), eventSlice.data(), filterSlice.data(), responseBuffer_[filterIndex].data());
                }
            }

            auto flowSlice = std::make_shared<FlowSlice>(slice->rows(), slice->cols());
            weightedResponseBufferX_.setZero();
            weightedResponseBufferY_.setZero();
            for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {

                float rad = deg2rad(this->angles_[filterIndex]);
                gpuAXPY(weightedResponseBufferX_.count(), std::cos(rad), responseBuffer_[filterIndex].data(), weightedResponseBufferX_.data());
                gpuAXPY(weightedResponseBufferY_.count(), -std::sin(rad), responseBuffer_[filterIndex].data(), weightedResponseBufferY_.data());
            }

            extractFilterResponse(weightedResponseBufferX_, flowSlice->xv_);
            extractFilterResponse(weightedResponseBufferY_, flowSlice->yv_);

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
    /**
     * Casts std::complex<float>* to thrust::complex<float>*, which are binary compatible
     * It is required due to CUDA not working with std::complex implementation.
     */
    thrust::complex<float>* castThrust(std::complex<float>* ptr) {
        LOG_FUN;
        return reinterpret_cast<thrust::complex<float>*>(ptr);
    }

    void extractFilterResponse(const ComplexBlob& deviceResponse, RealMatrix& hostResponse) {
        LOG_FUN_START;

        deviceResponse.copyTo(castThrust(transformedDataBuffer_.data()));
        this->transformer_->backward(transformedDataBuffer_, inversedDataBuffer_);
        this->padder_->extractDenseOutput(inversedDataBuffer_, hostResponse);

        LOG_FUN_END;
    }

private:
    RealMatrix paddedDataBuffer_;
    RealMatrix inversedDataBuffer_;
    ComplexMatrix transformedDataBuffer_;
    ComplexBlob weightedResponseBufferX_;
    ComplexBlob weightedResponseBufferY_;
    std::vector<std::vector<ComplexBlob>> filters_;
    std::vector<ComplexBlob> responseBuffer_;
    boost::circular_buffer<ComplexBlob> eventBuffer_;
    int rowsTransformed_;
    int colsTransformed_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINECPU_H
