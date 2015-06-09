//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_FILTERINGENGINE_H
#define OPTICAL_FLOW_FILTERINGENGINE_H

#include "common.h"
#include "DataFlowPolicy.h"
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
class FilteringEngine :
        public BufferedInputPolicy<EventSlice::Ptr, InputBufferT>,
        public BufferedOutputPolicy<FlowSlice::Ptr, OutputBufferT> {
public:
    using InputBuffer = typename BufferedInputPolicy<EventSlice::Ptr, InputBufferT>::InputBuffer;
    using OutputBuffer = typename BufferedOutputPolicy<FlowSlice::Ptr, OutputBufferT>::OutputBuffer;

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
              transformer_(std::move(transformer)) {

        LOG_FUN_START;

        factory_->setFilterTransformer(
                [this](const RealMatrix& filter) {

                    RealMatrix padded;
                    this->padder_->padFilter(filter, padded);
                    return padded;
                });

        LOG_FUN_END;
    }

    virtual void storeFilter(std::shared_ptr<Filter> filter) = 0;
    virtual void prepareResponseBuffer() = 0;
    virtual void initialize(std::shared_ptr<Filter> filter) = 0;
    virtual bool isInitialized()  = 0;
    /**
     * Returns number of filters;
     */
    virtual int numFilters() = 0;

    /**
     * Filter a single event slice
     * @param slice An event slices to be filetered.
     */
    virtual void filter(std::shared_ptr<EventSlice> slice) = 0;

    /**
     * Adds a filter with a specified angle.
     * @param angle Filter angle in degrees.
     */
    void addFilter(int angle) {
        LOG_FUN_START;
        LOG(INFO) << "Adding " << angle << " degree filter";

        auto it = std::find(std::begin(angles_), std::end(angles_), angle);

        if(it == angles_.end()) {
            auto filter = factory_->createFilter(angle);
            timeSteps_ = filter->numSlices();
            storeFilter(filter);
            prepareResponseBuffer();

            if(!isInitialized()) {
                initialize(filter);
            }
        }
        LOG_FUN_END;
    }

    void process() {
        LOG_FUN_START;
        while(this->hasInput()) {
            auto input = this->inputBuffer_->front();
            this->inputBuffer_->pop();
            filter(input);
        }
        LOG_FUN_END;
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
    bool isBufferFilled() {
        LOG_FUN;
        return timeSteps_ != 0 && receivedEventSlices_ >= static_cast<std::size_t>(timeSteps_);
    }

protected:
    int timeSteps_;
    size_t receivedEventSlices_;
    std::unique_ptr<IFilterFactory> factory_;
    std::unique_ptr<FourierPadder> padder_;
    std::unique_ptr<IFourierTransformer> transformer_;
    std::vector<int> angles_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINE_H
