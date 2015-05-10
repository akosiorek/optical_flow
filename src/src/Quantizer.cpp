//
// Created by Adam Kosiorek on 09.05.15.
//

#include <glog/logging.h>
#include "Quantizer.h"

Quantizer::Quantizer(int timeResolution)
    : initialized_(false), nextEventTime_(0), timeResolution_(timeResolution),
    eventSlices_(new std::deque<EventSlice>()) {
    currentEvents_.reserve(100);
};

void Quantizer::quantize(const std::vector<Event> &events) {

    if(events.empty()) {
        return;
    }

    if(!initialized_) {
        nextEventTime_ = events[0].time_ + timeResolution_;
        initialized_ = true;
    }

    for(const auto& event : events) {
        while(event.time_ >= nextEventTime_) {
            advanceTimeStep();
        }
        currentEvents_.emplace_back(event.x_, event.y_, event.parity_);
    }

    if(events[events.size() - 1].time_ == nextEventTime_ - 1) {
        advanceTimeStep();
    }
}

void Quantizer::advanceTimeStep() {
    if(!currentEvents_.empty()) {
        EventSlice slice;
        slice.setFromTriplets(currentEvents_.begin(), currentEvents_.end());
        currentEvents_.clear();
        eventSlices_->push_back(slice);
    } else {
        eventSlices_->emplace_back(EventSlice());
    }

    nextEventTime_ += timeResolution_;
}

std::shared_ptr<std::deque<EventSlice>> Quantizer::getEventSlices() {
    auto oldSlices = eventSlices_;
    eventSlices_.reset(new std::deque<EventSlice>());
    return oldSlices;
}

EventSlice Quantizer::getEventSlice() {
    if(eventSlices_->empty()) {
        return EventSlice();
    }
    auto slice = eventSlices_->front();
    eventSlices_->pop_front();
    return slice;
}

bool Quantizer::isEmpty() {
    return eventSlices_->empty();
}
