//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_FILTERINGENGINECPU_H
#define OPTICAL_FLOW_FILTERINGENGINECPU_H

#include <boost/circular_buffer.hpp>
#include "FilteringEngine.h"

template<template<class> class InputBufferT, template<class> class OutputBufferT = InputBufferT>
class FilteringEngineCPU : public FilteringEngine<InputBufferT, OutputBufferT> {
public:
    using BaseT = FilteringEngine<InputBufferT, OutputBufferT>;
    using BaseT::timeSteps_;

    /**
     * @brief Creates the FilteringEngine
     *
     * @param factory Factory used to create filters.
     * @param padder  Pre-configured fourier padder used for padding filters and event slices.
     * @param transformer   Transforms filters and event slices into the Fourier domain.
     */
    FilteringEngineCPU(std::unique_ptr<IFilterFactory> factory,
                    std::unique_ptr<FourierPadder> padder,
                    std::unique_ptr<IFourierTransformer> transformer)
            : BaseT(std::move(factory), std::move(padder), std::move(transformer)),
            eventBuffer_(0) {}


    virtual void storeFilter(std::shared_ptr<Filter> filter) override {

        this->angles_.push_back(filter->angle());
        this->filters_.emplace_back();
        auto& filterVec = this->filters_[this->filters_.size() - 1];
        filterVec.reserve(filter->numSlices());

        ComplexMatrix transformed;
        for(int i = 0; i < filter->numSlices(); ++i) {
            this->transformer_->forward(filter->at(i), transformed);
            filterVec.push_back(transformed);
        }
        filterRows_ = filterVec[0].rows();
        filterCols_ = filterVec[0].cols();
    }

    virtual void prepareResponseBuffer() override {
        responseBuffer_.emplace_back(filterRows_, filterCols_);
    }

    virtual bool isInitialized() override {
        return  !this->angles_.empty() && eventBuffer_.size() == timeSteps_;
    }

    virtual void initialize(std::shared_ptr<Filter> filter) override {
        // intialize buffer by allocating memory for all event slices to be kept
        if(eventBuffer_.size() != timeSteps_) {
            LOG(INFO) << "timeSteps_ " << timeSteps_ << " eventBuffer_.size() " <<  eventBuffer_.size() << " ...";
            eventBuffer_.set_capacity(timeSteps_);
            LOG(INFO) << " Now : timeSteps_ " << timeSteps_ << " eventBuffer_.size() " <<  eventBuffer_.size() << " ...";
            google::FlushLogFiles(google::GLOG_ERROR);
            while(eventBuffer_.size() != eventBuffer_.capacity()) {
                eventBuffer_.push_back(ComplexMatrix(filter->xSize(), filter->ySize()));
                LOG(INFO) << "timeSteps_ " << timeSteps_ << " eventBuffer_.size() " <<  eventBuffer_.size() << " ...";
            }
            extractedDataBuffer_.resize(filter->xSize(), filter->ySize());
        }
    }

    virtual void filter(std::shared_ptr<EventSlice> slice) override {
        LOG_FUN_START;

        eventBuffer_.rotate(eventBuffer_.end() - 1);
        ++this->receivedEventSlices_;
        this->padder_->padData(*slice, paddedDataBuffer_);
        this->transformer_->forward(paddedDataBuffer_, eventBuffer_[0]);

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
                    responseBuffer_[filterIndex] += eventSlice.cwiseProduct(filterSlice);
                }
            }



            auto flowSlice = std::make_shared<FlowSlice>(slice->rows(), slice->cols());
            for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {

                float rad = deg2rad(this->angles_[filterIndex]);
                this->transformer_->backward(responseBuffer_[filterIndex], inversedDataBuffer_);
                this->padder_->extractDenseOutput(inversedDataBuffer_, extractedDataBuffer_);
                flowSlice->xv_ += std::cos(rad) * extractedDataBuffer_ / this->transformer_->fftScaling_;
                flowSlice->yv_ -= std::sin(rad) * extractedDataBuffer_ / this->transformer_->fftScaling_;
            }
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
    std::vector<std::vector<ComplexMatrix>> filters_;
    std::vector<ComplexMatrix> responseBuffer_;
    boost::circular_buffer<ComplexMatrix> eventBuffer_;
    int filterRows_;
    int filterCols_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINECPU_H
