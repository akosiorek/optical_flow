#ifndef FLOW_SINK_PROCESSOR_H
#define FLOW_SINK_PROCESSOR_H

#include <vector>

#include "common.h"
#include "DataFlowPolicy.h"
#include "FlowSlice.h"
#include "IFlowSinkTask.h"

template<template <class> class InputBufferT>
class FlowSinkProcessor :
	public BufferedInputPolicy<FlowSlice::Ptr, InputBufferT>
{
public:
	using InputBuffer = typename BufferedInputPolicy<FlowSlice::Ptr, InputBufferT>::InputBuffer;

	FlowSinkProcessor()
		:	processThread_(nullptr),
			running_(false)
	{}

	~FlowSinkProcessor()
	{
		stop();
	}

	bool start()
	{
		LOG_FUN_START;

		processThread_ = std::make_unique<std::thread>(&FlowSinkProcessor::processFlowSlices, this);

		if(processThread_ != nullptr) return running_ = true, running_;
		else return false;

		LOG_FUN_END;
	}

	void stop()
	{
		if(running_==true)
		{
			running_ = false;
			processThread_->join();
			processThread_.reset();
		}
	}

	void addTask(std::unique_ptr<IFlowSinkTask> task)
	{
		taskQueue_.emplace_back(std::move(task));
	}

private:

	void processFlowSlices()
	{
		while(running_ == true)
		{
			if(this->hasInput())
			{
				auto input = this->inputBuffer_->front();
				this->inputBuffer_->pop();
				for(std::size_t i = 0; i<taskQueue_.size(); ++i)
				{
					taskQueue_[i]->process(input);
				}
			}
		}
	}

	std::vector<std::unique_ptr<IFlowSinkTask> > taskQueue_;

	std::unique_ptr<std::thread> processThread_;
	std::atomic_bool running_;
};

#endif