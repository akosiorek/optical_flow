//
// Created by Adam Kosiorek on 14.05.15.
//

#ifndef OPTICAL_FLOW_FILTER_H
#define OPTICAL_FLOW_FILTER_H

#include <vector>
#include <memory>
#include <Eigen/Core>

#include <glog/logging.h>

class Filter {
public:
    Filter() : angle_(0), xSize_(0), ySize_(0), filters_(nullptr) {}
    Filter(float angle, std::unique_ptr<std::vector<Eigen::MatrixXf>> filters)
    : angle_(angle), filters_(std::move(filters)) {

        CHECK_GT(filters_->size(), 0);
        xSize_ = at(0).cols();
        ySize_ = at(0).rows();
    }

    const Eigen::MatrixXf& at(int index) const {
        return filters_->at(index);
    }

    const Eigen::MatrixXf& operator[](int index) const {
        return at(index);
    }

    float angle() const {
        return angle_;
    }

    int numSlices() const {
        return filters_->size();
    }

    bool empty() const {
        return filters_ != nullptr && filters_->size() != 0;
    }

    int xSize() const {
        return xSize_;
    }

    int ySize() const {
        return ySize_;
    }

private:
    float angle_;
    int xSize_;
    int ySize_;
    std::unique_ptr<std::vector<Eigen::MatrixXf>> filters_;
};


#endif //OPTICAL_FLOW_FILTER_H
