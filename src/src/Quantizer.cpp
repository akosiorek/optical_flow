//
// Created by Adam Kosiorek on 09.05.15.
//

#include <glog/logging.h>
#include "Quantizer.h"

Quantizer::Quantizer(int timeResolution)
    : initialized_(false), nextEventTime_(0), timeResolution_(timeResolution) {
    currentSlice_.setZero();
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
            eventSlices_.push(currentSlice_);
            currentSlice_ = EventSlice();
            currentSlice_.setZero();
            nextEventTime_ += timeResolution_;
        }
        currentSlice_(event.x_, event.y_) = event.edge_;
    }

    if(events[events.size() - 1].time_ == nextEventTime_ - 1) {
        eventSlices_.push(currentSlice_);
        currentSlice_ = EventSlice();
        currentSlice_.setZero();
        nextEventTime_ += timeResolution_;
    }
}

std::vector<EventSlice> Quantizer::getEventSlices() {
    std::vector<EventSlice> events;
    while(!eventSlices_.empty()) {
        events.push_back(eventSlices_.front());
        eventSlices_.pop();
    }
    return events;
}

EventSlice Quantizer::getEventSlice() {
    if(eventSlices_.empty()) {
        EventSlice e;
        e.setZero();
        return e;
    }
    auto slice = eventSlices_.front();
    eventSlices_.pop();
    return slice;
}

bool Quantizer::isEmpty() {
    return eventSlices_.empty();
}

void Quantizer::init(int time) {
    eventSlices_.push(EventSlice());
    eventSlices_.back().setZero();
    nextEventTime_ = time;
}
