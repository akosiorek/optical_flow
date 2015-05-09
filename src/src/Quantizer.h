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
public:

    Quantizer(int timeResolution)
        : currentTimeStep_(0), timeResolution_(timeResolution)
         {};

    void quantize(const std::vector<Event>& events);
    EventSlice getEventSlice();
    std::vector<EventSlice> getEventSlices();


    unsigned int getTimeResolution() const {
        return timeResolution_;
    }

    int currentTimeStep_;

private:
    unsigned int timeResolution_;
    std::queue<EventSlice> eventSlices_;

};


#endif //NAME_QUANTIZER_H
