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

class EventSlice;

class Quantizer {
public:

    Quantizer(int timeResolution);

    void quantize(const std::vector<Edvs::Event>& events);
    bool isEmpty();
    EventSlice getEventSlice();
    std::shared_ptr<std::deque<EventSlice>> getEventSlices();


//  === Getters     ===========================================================
    unsigned int getTimeResolution() const {
        return timeResolution_;
    }

    EventTime getCurrentTimeStep() const {
        return nextEventTime_;
    }

private:
    void advanceTimeStep();

private:
    bool initialized_;
    EventTime nextEventTime_;
    unsigned int timeResolution_;
    std::shared_ptr<std::deque<EventSlice>> eventSlices_;
    std::vector<Eigen::Triplet<int>> currentEvents_;
};


#endif //NAME_QUANTIZER_H
