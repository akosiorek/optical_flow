//
// Created by Adam Kosiorek on 14.05.15.
//

#include "Filter.h"


Filter::Filter() : angle_(0), xSize_(0), ySize_(0), filters_(nullptr) {}
Filter::Filter(float angle, std::unique_ptr<std::vector<FilterT>> filters)
        : angle_(angle), filters_(std::move(filters)) {

    if(filters_->size() == 0) {
        THROW_INVALID_ARG("There has to be at least one filter");
    }
    xSize_ = at(0).cols();
    ySize_ = at(0).rows();
}

auto Filter::at(int index) const -> const FilterT& {
    return filters_->at(index);
}

auto Filter::operator[](int index) const -> const FilterT& {
    return at(index);
}

float Filter::angle() const {
    return angle_;
}

int Filter::numSlices() const {
    return filters_->size();
}

bool Filter::empty() const {
    return filters_ != nullptr && filters_->size() != 0;
}

int Filter::xSize() const {
    return xSize_;
}

int Filter::ySize() const {
    return ySize_;
}