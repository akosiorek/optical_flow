//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_FILTERINGENGINECPU_H
#define OPTICAL_FLOW_FILTERINGENGINECPU_H

#include "arrayfire.h"
#include <boost/circular_buffer.hpp>
#include "FilteringEngine.h"

template<template<class> class InputBufferT, template<class> class OutputBufferT = InputBufferT>
class FilteringEngineGPU : public FilteringEngine<InputBufferT, OutputBufferT> {
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
    FilteringEngineGPU(std::unique_ptr<IFilterFactory> factory,
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

            filterVec.emplace_back(transformed.rows(), transformed.cols(), reinterpret_cast<af::cfloat*>(transformed.data()));
        }
        filterRows_ = transformed.rows();
        filterCols_ = transformed.cols();
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

            while(eventBuffer_.size() != eventBuffer_.capacity()) {
                eventBuffer_.push_back(af::array(filterRows_, filterCols_, c32));
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
        this->transformer_->forward(paddedDataBuffer_, transformedDataBuffer_);
        eventBuffer_[0] = af::array(eventBuffer_[0].dims(), reinterpret_cast<af::cfloat*>(transformedDataBuffer_.data()));

        // may the magic happen
        if(this->isBufferFilled()) {

            // zero the response buffers
            for(auto& buffer : responseBuffer_) {
                buffer = af::constant(0, buffer.dims());
            }

            // iterate over eventSlices and filterSlices
            for(int sliceIndex = 0; sliceIndex < timeSteps_; ++sliceIndex) {
                const auto& eventSlice = eventBuffer_[sliceIndex];
                // iterate over filters
                for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {
                    const auto& filterSlice = filters_[filterIndex][sliceIndex];

                    LOG(ERROR) << responseBuffer_[filterIndex].dims();
                    LOG(ERROR) << eventSlice.dims();
                    LOG(ERROR) << filterSlice.dims();

                    responseBuffer_[filterIndex] += eventSlice * filterSlice;
                }
            }



            auto flowSlice = std::make_shared<FlowSlice>(slice->rows(), slice->cols());
            for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {

                float rad = deg2rad(this->angles_[filterIndex]);
                std::complex<float>* data = reinterpret_cast<std::complex<float>*>(responseBuffer_[filterIndex].host<af::cfloat>());
                auto dims = responseBuffer_[filterIndex].dims();
                Eigen::Map<ComplexMatrix> mappedData(data, dims[0], dims[1]);
                this->transformer_->backward(mappedData, inversedDataBuffer_);
                delete[] data;

                this->padder_->extractDenseOutput(inversedDataBuffer_, extractedDataBuffer_);
                flowSlice->xv_ += std::cos(rad) * extractedDataBuffer_;
                flowSlice->yv_ -= std::sin(rad) * extractedDataBuffer_;
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
    ComplexMatrix transformedDataBuffer_;
    std::vector<std::vector<af::array>> filters_;
    std::vector<af::array> responseBuffer_;
    boost::circular_buffer<af::array> eventBuffer_;
    int filterRows_;
    int filterCols_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINECPU_H
