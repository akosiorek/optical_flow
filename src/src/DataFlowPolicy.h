//
// Created by Adam Kosiorek on 6/8/15.
//

#ifndef OPTICAL_FLOW_DATAFLOWPOLICY_H
#define OPTICAL_FLOW_DATAFLOWPOLICY_H

#include <memory>

template<class Input, template<class> class InputBufferT>
struct BufferedInputPolicy {
    using InputBuffer = InputBufferT<Input>;

    BufferedInputPolicy(std::shared_ptr<InputBuffer> inputBuffer)
            : inputBuffer_(inputBuffer) {
        CHECK(inputBuffer) << "InputBuffer cannot be null.";
    }

    BufferedInputPolicy() : BufferedInputPolicy(std::make_shared<InputBuffer>()) {}

    bool hasInput() {
        return inputBuffer_ && !inputBuffer_->empty();
    }

    void setInputBuffer(std::shared_ptr<InputBuffer> inputBuffer) {
        this->inputBuffer_ = inputBuffer;
    }

    std::shared_ptr<InputBuffer> getInputBuffer() {
        return inputBuffer_;
    }

protected:
    std::shared_ptr<InputBuffer> inputBuffer_;
};

template<class Output, template<class> class OutputBufferT>
struct BufferedOutputPolicy {
    using OutputBuffer = OutputBufferT<Output>;

    BufferedOutputPolicy(std::shared_ptr<OutputBuffer> outputBuffer)
            : outputBuffer_(outputBuffer) {
        CHECK(outputBuffer) << "OutputBuffer cannot be null.";
    }

    BufferedOutputPolicy() : BufferedOutputPolicy(std::make_shared<OutputBuffer>()) {}

    bool hasOutput() {
        return outputBuffer_ && !outputBuffer_->empty();
    }

    void setOutputBuffer(std::shared_ptr<OutputBuffer> outputBuffer) {
        this->outputBuffer_ = outputBuffer;
    }

    std::shared_ptr<OutputBuffer> getOutputBuffer() {
        return outputBuffer_;
    }

protected:
    std::shared_ptr<OutputBuffer> outputBuffer_;
};

#endif //OPTICAL_FLOW_DATAFLOWPOLICY_H
