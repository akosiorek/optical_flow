//
// Created by adam on 09.05.15.
//

#ifndef NAME_QUANTIZER_H
#define NAME_QUANTIZER_H

#include <deque>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "Edvs/Event.hpp"

#include "common.h"
#include "types.h"
#include "EventSlice.h"

template < template<typename> class BufferType>
class Quantizer {
public:

    using EventQueueT = BufferType<Event>;
    using EventSliceQueueT = BufferType<EventSlice::Ptr>;;

    Quantizer(int timeResolution)
    :   initialized_(false),
        nextEventTime_(0),
        timeResolution_(timeResolution),
        inputBuffer_(new EventQueueT()),
        outputBuffer_(new EventSliceQueueT())
    {
        LOG_FUN_START;
        currentEvents_.reserve(100);
        LOG_FUN_END;
    };

    void process()
    {
        LOG_FUN_START;
        std::vector<Event> events;
        while(hasInput())
        {
            events.push_back(inputBuffer_->front());
            inputBuffer_->pop();
        }
        quantize(events);
        LOG_FUN_END;
    }

    /*
     * Checks if the input buffer is not empty.
     */
    bool hasInput() {
        return inputBuffer_ && !inputBuffer_->empty();
    }

    void quantize(const std::vector<Edvs::Event>& events)
    {
        LOG_FUN_START;
        if(events.empty()) {
            return;
        }

        if(!initialized_) {
            nextEventTime_ = events[0].t + timeResolution_;
            initialized_ = true;
        }

        for(const auto& event : events) {
            while(event.t >= nextEventTime_) {
                advanceTimeStep();
            }
            currentEvents_.emplace_back(event.x, event.y, (event.parity!=0) ? 1 : -1);
        }

        if(events[events.size() - 1].t == nextEventTime_ - 1) {
            advanceTimeStep();
        }
        LOG_FUN_END;
    }

    bool isEmpty()
    {
        LOG_FUN;
        return outputBuffer_->empty();
    }


    EventSlice getEventSlice()
    {
        LOG_FUN_START;
        if(outputBuffer_->empty()) {
            return EventSlice();
        }

        auto slice = outputBuffer_->front();
        outputBuffer_->pop();
        LOG_FUN_END;
        return *slice;
    }

    std::shared_ptr<EventSliceQueueT> getEventSlices()
    {
        LOG_FUN_START;
        auto oldSlices = outputBuffer_;
        outputBuffer_.reset(new EventSliceQueueT());
        LOG_FUN_END;
        return oldSlices;
    }

    void setInputBuffer(std::shared_ptr<EventQueueT> buffer) {
        LOG_FUN;
        this->inputBuffer_ = buffer;
    }

    void setOutputBuffer(std::shared_ptr<EventSliceQueueT> buffer) {
        LOG_FUN;
        this->outputBuffer_ = buffer;
    }

//  === Getters     ===========================================================
    unsigned int getTimeResolution() const {
        LOG_FUN;
        return timeResolution_;
    }

    EventTime getCurrentTimeStep() const {
        LOG_FUN;
        return nextEventTime_;
    }

private:
    void advanceTimeStep()
    {
        LOG_FUN_START;
        if(!currentEvents_.empty()) {
            EventSlice::Ptr slice(new EventSlice);
            slice->setFromTriplets(currentEvents_.begin(), currentEvents_.end());
            currentEvents_.clear();
            outputBuffer_->push(slice);
        } else {
            outputBuffer_->emplace(std::make_shared<EventSlice>());
        }

        nextEventTime_ += timeResolution_;
        LOG_FUN_END;
    }

private:
    bool initialized_;
    EventTime nextEventTime_;
    unsigned int timeResolution_;
    std::shared_ptr<EventQueueT> inputBuffer_;
    std::shared_ptr<EventSliceQueueT> outputBuffer_;
    std::vector<Eigen::Triplet<int>> currentEvents_;
};


#endif //NAME_QUANTIZER_H
