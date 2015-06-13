//
// Created by Adam Kosiorek on 6/12/15.
//

#ifndef OPTICAL_FLOW_DEVICEBLOB_H
#define OPTICAL_FLOW_DEVICEBLOB_H

#include <cstdlib>

template<class Dtype>
class DeviceBlob {
public:
    DeviceBlob();
    DeviceBlob(int rows, int cols);
    DeviceBlob(int rows, int cols, const Dtype* from);
    DeviceBlob(const DeviceBlob& that);
    ~DeviceBlob();

    void copyFrom(const Dtype* from);
    void copyTo(Dtype* to) const;
    void setZero();
    size_t rows() const;
    size_t cols() const;
    size_t count() const;
    Dtype* data();
    const Dtype* data() const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t count_;
    std::size_t bytes_;
    Dtype* data_;
};

#endif //OPTICAL_FLOW_DEVICEBLOB_H
