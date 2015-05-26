//
// Created by Adam Kosiorek on 22.05.15.
//

#ifndef OPTICAL_FLOW_FLOWSLICE_H
#define OPTICAL_FLOW_FLOWSLICE_H

#include <Eigen/Core>

class FlowSlice {
public:
    FlowSlice(int xSize, int ySize) : xv_(xSize, ySize), yv_(xSize, ySize) {
        xv_.setZero();
        yv_.setZero();
    }

    FlowSlice() : FlowSlice(128, 128) {}

    Eigen::MatrixXf xv_;
    Eigen::MatrixXf yv_;
};

#endif //OPTICAL_FLOW_FLOWSLICE_H
