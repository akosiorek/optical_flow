#ifndef I_FLOW_SINK_TASK_H
#define I_FLOW_SINK_TASK_H

class FlowSlice;

class IFlowSinkTask
{
public:
	virtual ~IFlowSinkTask() = 0;
	virtual void process(FlowSlice& flowSlice);
};

#endif //I_FLOW_SINK_TASK_H