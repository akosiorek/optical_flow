//
// Created by adam on 09.05.15.
//

#ifndef NAME_QUANTIZER_H
#define NAME_QUANTIZER_H

#include <deque>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "Event.h"
#include "common.h"

class EventSlice : public Eigen::SparseMatrix<int> {
public:
    EventSlice() : Eigen::SparseMatrix<int>(128, 128) {};

    int& operator()(int x, int y) {
        return this->coeffRef(y, x);
    }

    bool isZero() const {
        return this->nonZeros() == 0 || this->squaredNorm() < 1e-8;
    }
};

class Quantizer {
public:

    Quantizer(int timeResolution);

    void quantize(const std::vector<Event>& events);
    bool isEmpty();
    EventSlice getEventSlice();
    std::shared_ptr<std::deque<EventSlice>> getEventSlices();


//  === Getters     ===========================================================
    unsigned int getTimeResolution() const {
        return timeResolution_;
    }

    Event::TimeT getCurrentTimeStep() const {
        return nextEventTime_;
    }

private:
    void advanceTimeStep();

private:
    bool initialized_;
    Event::TimeT nextEventTime_;
    unsigned int timeResolution_;
    std::shared_ptr<std::deque<EventSlice>> eventSlices_;
    std::vector<Eigen::Triplet<int>> currentEvents_;
};


#endif //NAME_QUANTIZER_H
