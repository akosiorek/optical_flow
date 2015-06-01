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

template < template<typename> typename BufferType>
class Quantizer {
public:

    Quantizer(int timeResolution)
    :   initialized_(false), 
        nextEventTime_(0), 
        timeResolution_(timeResolution),
        eventSlices_(new BufferType<EventSlice>()) 
    {
        currentEvents_.reserve(100);
    };

    void quantize(const std::vector<Edvs::Event>& events)
    {
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
    }

    bool isEmpty()
    {
        return eventSlices_->empty();
    }


    EventSlice getEventSlice()
    {
        if(eventSlices_->empty()) {
            return EventSlice();
        }

        auto slice = eventSlices_->front();
        eventSlices_->pop();
        return slice;
    }

    std::shared_ptr<BufferType<EventSlice>> getEventSlices()
    {
        auto oldSlices = eventSlices_;
        eventSlices_.reset(new BufferType<EventSlice>());
        return oldSlices;
    }


//  === Getters     ===========================================================
    unsigned int getTimeResolution() const {
        return timeResolution_;
    }

    EventTime getCurrentTimeStep() const {
        return nextEventTime_;
    }

private:
    void advanceTimeStep()
    {
        if(!currentEvents_.empty()) {
            EventSlice slice;
            slice.setFromTriplets(currentEvents_.begin(), currentEvents_.end());
            currentEvents_.clear();
            eventSlices_->push(slice);
        } else {
            eventSlices_->emplace(EventSlice());
        }

        nextEventTime_ += timeResolution_;
    }

private:
    bool initialized_;
    EventTime nextEventTime_;
    unsigned int timeResolution_;
    std::shared_ptr<BufferType<EventSlice>> eventSlices_;
    std::vector<Eigen::Triplet<int>> currentEvents_;
};


#endif //NAME_QUANTIZER_H
