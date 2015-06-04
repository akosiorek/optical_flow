//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_FILTERINGENGINE_H
#define OPTICAL_FLOW_FILTERINGENGINE_H

#include <memory>

#include <boost/circular_buffer.hpp>

#include "EventSlice.h"
#include "FlowSlice.h"
#include "IFilterFactory.h"
#include "FourierPadder.h"
#include "IFourierTransformer.h"

/**
 * @brief Computes optical flow.
 *
 * @param InputBufferT input buffer template type, will store incoming event slices.
 * @param OutputBufferT output buffer template type, will store outgoing flow slices.
 */
template<template <class> class InputBufferT, template <class> class OutputBufferT = InputBufferT>
class FilteringEngine {
public:
    using EventQueueT = InputBufferT<std::shared_ptr<EventSlice>>;
    using FlowQueueT = OutputBufferT<std::shared_ptr<FlowSlice>>;

    /**
     * @brief Creates the FilteringEngine
     *
     * @param factory Factory used to create filters.
     * @param padder  Pre-configured fourier padder used for padding filters and event slices.
     * @param transformer   Transforms filters and event slices into the Fourier domain.
     */
    FilteringEngine(std::unique_ptr<IFilterFactory> factory,
                    std::unique_ptr<FourierPadder> padder,
                    std::unique_ptr<IFourierTransformer> transformer)

            : timeSteps_(0),
              receivedEventSlices_(0),
              factory_(std::move(factory)),
              padder_(std::move(padder)),
              transformer_(std::move(transformer)),
              eventBuffer_(0) {

        LOG_FUN_START;

        factory_->setFilterTransformer(
                [this](const RealMatrix& filter) {

                    RealMatrix padded;
                    this->padder_->padFilter(filter, padded);
                    ComplexMatrix transformed(padded.rows(),padded.cols()/2 + 1);
                    this->transformer_->forward(padded, transformed);
                    return transformed;
                });

        LOG_FUN_END;
    }

    /**
     * Adds a filter with a specified angle.
     * @param angle Filter angle in degrees.
     */
    void addFilter(int angle) {
        LOG_FUN_START;
        LOG(INFO) << "Adding " << angle << " degree filter";

        auto it = std::find_if(std::begin(filters_), std::end(filters_),
        [angle](std::shared_ptr<Filter> filter) {
            return filter->angle() == angle;
        });

        if(it == filters_.end()) {
            auto filter = factory_->createFilter(angle);

            filters_.push_back(filter);
            timeSteps_ = filter->numSlices();
            responseBuffer_.emplace_back(filter->ySize(), filter->xSize());
            LOG(INFO) << "Creating filter with angle " << angle << " xSize " <<  filter->xSize() <<  " ySize " << filter->ySize();

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

        LOG_FUN_END;
    }

    /**
     * Filter a single event slice
     * @param slice An event slices to be filetered.
     */
    void filter(std::shared_ptr<EventSlice> slice) {
        LOG_FUN_START;

        eventBuffer_.rotate(eventBuffer_.end() - 1);
        ++receivedEventSlices_;
        padder_->padData(*slice, paddedDataBuffer_);
        transformer_->forward(paddedDataBuffer_, eventBuffer_[0]);

        // may the magic happen
        if(isInitialized()) {

            // zero the response buffers
            for(auto& buffer : responseBuffer_) {
                buffer.setZero();
            }

            // iterate over eventSlices and filterSlices
            for(int sliceIndex = 0; sliceIndex < timeSteps_; ++sliceIndex) {
                const auto& eventSlice = eventBuffer_[sliceIndex];
                // iterate over filters
                for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {
                    const auto& filterSlice = filters_[filterIndex]->at(sliceIndex);
                    responseBuffer_[filterIndex] += eventSlice.cwiseProduct(filterSlice);
                }
            }

            auto flowSlice = std::make_shared<FlowSlice>(slice->rows(), slice->cols());
            for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {

                float rad = deg2rad(filters_[filterIndex]->angle());
                transformer_->backward(responseBuffer_[filterIndex], inversedDataBuffer_);
                padder_->extractDenseOutput(inversedDataBuffer_, extractedDataBuffer_);
                flowSlice->xv_ += std::cos(rad) * extractedDataBuffer_;
                flowSlice->yv_ -= std::sin(rad) * extractedDataBuffer_;
            }
            outputBuffer_->push(flowSlice);
        }

        LOG_FUN_END;
    }

    void process() {
        LOG_FUN_START;
        while(hasInput()) {
            auto input = inputBuffer_->front();
            inputBuffer_->pop();
            filter(input);
        }
        LOG_FUN_END;
    }

    void setInputBuffer(std::shared_ptr<EventQueueT> buffer) {
        LOG_FUN;
        this->inputBuffer_ = buffer;
    }
    void setOutputBuffer(std::shared_ptr<FlowQueueT> buffer) {
        LOG_FUN;
        this->outputBuffer_ = buffer;
    }

    /**
     * Filter depth or the number of the filter slices.
     */
    int timeSteps() {
        LOG_FUN;
        return timeSteps_;
    }

    /**
     * Checks whether it has at least one filter and received a sufficient
     * number of events to start estimating flow
     */
    bool isInitialized() {
        LOG_FUN;
        return timeSteps_ != 0 && receivedEventSlices_ >= timeSteps_;
    }

    /*
     * Checks if the input buffer is not empty.
     */
    bool hasInput() {
        LOG_FUN;
        return inputBuffer_ && !inputBuffer_->empty();
    }

    /**
     * Checks if the output buffer is not empty.
     */
    bool hasOutput() {
        LOG_FUN;
        return outputBuffer_ && !outputBuffer_->empty();
    }

    /**
     * Returns number of filters;
     */
    int numFilters() {
        LOG_FUN;
        return filters_.size();
    }

private:
    int timeSteps_;
    size_t receivedEventSlices_;
    RealMatrix paddedDataBuffer_;
    RealMatrix extractedDataBuffer_;
    RealMatrix inversedDataBuffer_;
    std::shared_ptr<EventQueueT> inputBuffer_;
    std::shared_ptr<FlowQueueT> outputBuffer_;
    std::unique_ptr<IFilterFactory> factory_;
    std::unique_ptr<FourierPadder> padder_;
    std::unique_ptr<IFourierTransformer> transformer_;
    std::vector<std::shared_ptr<Filter>> filters_;
    std::vector<ComplexMatrix> responseBuffer_;
    boost::circular_buffer<ComplexMatrix> eventBuffer_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINE_H
