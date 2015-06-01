//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_EVENTSLICE_H
#define OPTICAL_FLOW_EVENTSLICE_H

#include "common.h"

class EventSlice : public SparseMatrix {
public:
    using Ptr = std::shared_ptr<EventSlice>;

    EventSlice(int xSize, int ySize) : SparseMatrix(ySize, xSize) {};
    EventSlice() : EventSlice(128, 128) {};

    float& operator()(int x, int y) {
        return this->at(x, y);
    }

    float& at(int x, int y) {
        return this->coeffRef(y, x);
    }

    bool isZero() const {
        return this->nonZeros() == 0 || this->squaredNorm() < 1e-8;
    }
};

#endif //OPTICAL_FLOW_EVENTSLICE_H
