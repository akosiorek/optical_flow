//
// Created by Adam Kosiorek on 14.05.15.
//

#ifndef OPTICAL_FLOW_FILTER_H
#define OPTICAL_FLOW_FILTER_H

#include <vector>
#include "common.h"

class Filter {
public:
    using MatrixT = RealMatrix;
    using FilterT = ComplexMatrix;

    Filter();
    Filter(float angle, std::unique_ptr<std::vector<FilterT>> filters);

    const FilterT& at(int index) const;
    const FilterT& operator[](int index) const;

    float angle() const;
    int numSlices() const;
    bool empty() const;
    int xSize() const;
    int ySize() const;

private:
    float angle_;
    int xSize_;
    int ySize_;
    std::unique_ptr<std::vector<FilterT>> filters_;
};


#endif //OPTICAL_FLOW_FILTER_H
