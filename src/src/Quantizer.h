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
#include "DataFlowPolicy.h"
#include "EventSlice.h"

template < template<class> class BufferT>
class Quantizer :
        public BufferedInputPolicy<Event, BufferT>,
        public BufferedOutputPolicy<EventSlice::Ptr, BufferT>{
public:
    using InputBuffer = typename BufferedInputPolicy<Event, BufferT>::InputBuffer;
    using OutputBuffer = typename BufferedOutputPolicy<EventSlice::Ptr, BufferT>::OutputBuffer;

    Quantizer(int timeResolution)
    :   initialized_(false),
        nextEventTime_(0),
        timeResolution_(timeResolution) {

        LOG_FUN_START;
        currentEvents_.reserve(100);
        LOG_FUN_END;
    };

    void process()
    {
        LOG_FUN_START;
        std::vector<Event> events;
        while(this->hasInput())
        {
            // events.push_back(inputBuffer_->front());
            this->inputBuffer_->pop(events);
        }
        quantize(events);
        LOG_FUN_END;
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
        return this->outputBuffer_->empty();
    }


    std::shared_ptr<EventSlice> getEventSlice()
    {
        LOG_FUN_START;
        if(this->outputBuffer_->empty()) {
            return std::make_shared<EventSlice>();
        }

        auto slice = this->outputBuffer_->front();
        this->outputBuffer_->pop();
        LOG_FUN_END;
        return slice;
    }

    std::shared_ptr<OutputBuffer> getEventSlices()
    {
        LOG_FUN_START;
        auto oldSlices = this->outputBuffer_;
        this->outputBuffer_.reset(new OutputBuffer());
        LOG_FUN_END;
        return oldSlices;
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
            this->outputBuffer_->push(slice);
        } else {
            this->outputBuffer_->emplace(std::make_shared<EventSlice>());
        }

        nextEventTime_ += timeResolution_;
        LOG_FUN_END;
    }

private:
    bool initialized_;
    EventTime nextEventTime_;
    unsigned int timeResolution_;
    std::vector<Eigen::Triplet<int>> currentEvents_;
};


#endif //NAME_QUANTIZER_H
