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
    typedef InputBufferT<std::shared_ptr<EventSlice>> EventQueueT;
    typedef OutputBufferT<std::shared_ptr<FlowSlice>> FlowQueueT;

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

            : factory_(std::move(factory)),
              padder_(std::move(padder)),
              transformer_(std::move(transformer)),
              timeSteps_(0),
              eventBuffer_(0) {

//        //TODO wait for dynamic reference-argument-accepting FourierPadder
//        factory->setFilterTransformer(
//                [this](const Eigen::MatrixXf& filter) {
//
//                    return this->padder_->padData(filter);
//                });
    }

    void setInputBuffer(std::shared_ptr<EventQueueT> buffer) {
        this->inputBuffer_ = buffer;
    }
    void setOutputBuffer(std::shared_ptr<FlowQueueT> buffer) {
        this->outputBuffer_ = buffer;
    }

    /**
     * Filter depth or the number of the filter slices.
     */
    int timeSteps() {
        return timeSteps_;
    }

    /**
     * Checks whether it has at least one filter and received a sufficient
     * number of events to start estimating flow
     */
    bool isInitialized() {
        return timeSteps_ != 0 && eventBuffer_.size() >= timeSteps_;
    }

    /*
     * Checks if the input buffer is not empty.
     */
    bool hasInput() {
        return inputBuffer_ && !inputBuffer_->empty();
    }

    /**
     * Checks if the output buffer is not empty.
     */
    bool hasOutput() {
        return outputBuffer_ && !outputBuffer_->empty();
    }

    /**
     * Adds a filter with a specified angle.
     * @param angle Filter angle in degrees.
     */
    void addFilter(int angle) {
        auto it = std::find_if(std::begin(filters_), std::end(filters_),
        [angle](std::shared_ptr<Filter> filter) {
            return filter->angle() == angle;
        });

        if(it == filters_.end()) {
            auto filter = factory_->createFilter(angle);
            filters_.push_back(filter);
            timeSteps_ = filter->numSlices();
            responseTemplates_.emplace_back(filter->xSize(), filter->ySize());
            responseTemplates_[responseTemplates_.size() - 1].setZero();

            if(eventBuffer_.size() != timeSteps_) {
                eventBuffer_.resize(timeSteps_);
            }
        }
    };

    /**
     * Filter a single event slice
     * @param slice An event slices to be filetered.
     */
    void filter(std::shared_ptr<EventSlice> slice) {

//        bool wasInitialized = isInitialized();
//        transformAndEnqueue(slice);
//
//        // may the magic happen
//        if(wasInitialized) {
//
//            eventStream_.pop_front();
//
//            int sliceIndex = 0;
//            auto eventIt = eventStream_.begin();
//            for(; eventIt != eventStream_.end(); ++sliceIndex, ++eventIt) {
//                const auto& eventSlice = **eventIt;
//                for(int filterIndex = 0; filterIndex < filters_.size(); ++filterIndex) {
//
//                    // iterate over event slices and filters/filter slices transforming,
//                    // multiplying, summing
//                }
//            }
//            // reverse transform
//            // weight by filter angles
//            // sum up
//            // put the result to the output queue
//        }
    }

private:
    /**
     * Transforms an event slice and enqueues it in the buffer.
     */
    void transformAndEnqueue(std::shared_ptr<EventSlice>& eventSlice) {
//        eventStream_.emplace_back();

        auto padded = padder_->padData(static_cast<const Eigen::SparseMatrix<float>&>(*eventSlice));
    }

private:
    std::shared_ptr<EventQueueT> inputBuffer_;
    std::shared_ptr<FlowQueueT> outputBuffer_;

    std::unique_ptr<IFilterFactory> factory_;
    std::unique_ptr<FourierPadder> padder_;
    std::unique_ptr<IFourierTransformer> transformer_;
    int timeSteps_;

    std::vector<std::shared_ptr<Filter>> filters_;
    std::vector<Eigen::MatrixXf> responseTemplates_;

    boost::circular_buffer<FourierPadder::EBOFMatrix> eventBuffer_;
};


#endif //OPTICAL_FLOW_FILTERINGENGINE_H
