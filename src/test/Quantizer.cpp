//
// Created by Adam Kosiorek on 09.05.15.
//

#include "Quantizer.h"

void Quantizer::quantize(const std::vector<Event> &events) {

    if(events.empty()) {
        return;
    }

    if(eventSlices_.empty()) {
        eventSlices_.push(EventSlice());
    }

    for(const auto& event : events) {
        if(event.time - currentTimeStep_ > timeResolution_) {
            currentTimeStep_ = event.time;
            eventSlices_.push(EventSlice());
        }
        eventSlices_.back()(event.x, event.y) += event.edge;
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
        throw std::runtime_error("No Events");
    }
    auto slice = eventSlices_.front();
    eventSlices_.pop();
    return slice;
}
