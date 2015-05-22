//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_EVENTSLICE_H
#define OPTICAL_FLOW_EVENTSLICE_H

#include <Eigen/SparseCore>

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

#endif //OPTICAL_FLOW_EVENTSLICE_H
