//
// Created by adam on 09.05.15.
//

#ifndef NAME_QUANTIZER_H
#define NAME_QUANTIZER_H

#include <queue>
#include <vector>
#include <Eigen/Dense>

#include "Event.h"

typedef Eigen::Matrix<int, 128, 128> EventSlice;

class Quantizer {
    typedef decltype(Event().time_) TimeType;
public:

    Quantizer(int timeResolution);

    void quantize(const std::vector<Event>& events);
    bool isEmpty();
    EventSlice getEventSlice();
    std::vector<EventSlice> getEventSlices();


//  === Getters     ===========================================================
    unsigned int getTimeResolution() const {
        return timeResolution_;
    }

    TimeType getCurrentTimeStep() const {
        return nextEventTime_;
    }

private:
    void init(int time);

private:
    bool initialized_;
    TimeType nextEventTime_;
    unsigned int timeResolution_;
    std::queue<EventSlice> eventSlices_;
    EventSlice currentSlice_;
};


#endif //NAME_QUANTIZER_H
