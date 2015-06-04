//
// Created by Adam Kosiorek on 14.05.15.
//

#include "Filter.h"


Filter::Filter() : angle_(0), xSize_(0), ySize_(0), filters_(nullptr) {}
Filter::Filter(float angle, std::unique_ptr<std::vector<FilterT>> filters)
        : angle_(angle), filters_(std::move(filters)) {

    LOG_FUN_START;

    if(filters_->size() == 0) {
        THROW_INVALID_ARG("There has to be at least one filter");
    }
    xSize_ = at(0).cols();
    ySize_ = at(0).rows();

    LOG_FUN_END;
}

auto Filter::at(int index) const -> const FilterT& {
    LOG_FUN;
    return filters_->at(index);
}

auto Filter::operator[](int index) const -> const FilterT& {
    LOG_FUN;
    return at(index);
}

float Filter::angle() const {
    LOG_FUN;
    return angle_;
}

int Filter::numSlices() const {
    LOG_FUN;
    return filters_->size();
}

bool Filter::empty() const {
    LOG_FUN;
    return filters_ != nullptr && filters_->size() != 0;
}

int Filter::xSize() const {
    LOG_FUN;
    return xSize_;
}

int Filter::ySize() const {
    LOG_FUN;
    return ySize_;
}