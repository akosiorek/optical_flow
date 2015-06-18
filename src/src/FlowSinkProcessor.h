#ifndef FLOW_SINK_PROCESSOR_H
#define FLOW_SINK_PROCESSOR_H

#include <queue>

#include "common.h"
#include "DataFlowPolicy.h"
#include "IFlowSinkTask.h"

class FlowSinkProcessor : public BufferedInputPolicy<EventSlice::Ptr, InputBufferT>
{
public:

	FlowSinkProcessor()
		: running_(false),
			processThread_(nullptr)
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
		running_ = false;
		processThread_->join();
		processThread_.reset();
	}

	void addTask(std::unique_ptr<IFlowSinkTask> task)
	{
		taskQueue_.push(task);
	}

private:

	void processFlowSlices()
	{
		while(running_ == true)
		{
			for(auto task : taskQueue_)
			{
				task->process();
			}
		}
	}

	std::queue<std::unique_ptr<IFlowSinkTask> > taskQueue_;

	std::unique_ptr<std::thread> processThread_;
	std::atomic_bool running_;
};

#endif