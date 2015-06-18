#ifndef TASK_WRITE_FLOW_SLICE_H
#define TASK_WRITE_FLOW_SLICE_H

#include "IFlowSinkTask.h"
#include "FlowSlice.h"

class TaskWriteFlowSlice : public IFlowSinkTask
{

public:
	void process(FlowSlice& flowSlice) overwrite
	{

	};
};

#endif