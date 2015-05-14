//
// Created by Adam Kosiorek on 14.05.15.
//

#ifndef OPTICAL_FLOW_FILTER_H
#define OPTICAL_FLOW_FILTER_H

#include <vector>
#include <memory>
#include <Eigen/Core>

class Filter {
public:
    Filter();
    Filter(float angle, std::unique_ptr<std::vector<Eigen::MatrixXf>> filters)
    : angle_(angle), filters_(std::move(filters)) {};

    const Eigen::MatrixXf& at(int index) const {
        return filters_->at(index);
    }

    const Eigen::MatrixXf& operator[](int index) const {
        return at(index);
    }

    float getAngle_() const {
        return angle_;
    }

private:
    float angle_;
    std::unique_ptr<std::vector<Eigen::MatrixXf>> filters_;
};


#endif //OPTICAL_FLOW_FILTER_H
