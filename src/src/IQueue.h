#ifndef I_QUEUE_H
#define I_QUEUE_H

/**
 * @brief A very basic interface wrapper for STL compliant containers offering queue-like access.
 * 		  Thread-safety is up to the implementation.
 */
template<typename T, typename Queue >
class IQueue
{
public:
	virtual~IQueue() = default;

	// Element Access
	virtual T& front() = 0;
	virtual const T& front() const = 0;

	virtual T& back() = 0;
	virtual const T& back() const = 0;

	// Modifiers
	virtual void push(const T& val) = 0;
	virtual void pushVector(const std::vector<T>& vals) = 0;

	// Some might a emplace here...

	virtual void pop() = 0;
	virtual void swap(Queue& x) = 0;
	virtual void clear();

	// Combined Modifiers (less locks)
	virtual void pop(T& val) = 0;
	virtual void pop(std::vector<T>& vals) = 0;

	// Capacity
	virtual bool empty() const = 0;
	virtual std::size_t size() const = 0;
};

#endif //I_QUEUE_H
