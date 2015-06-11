#ifndef I_QUEUE_H
#define I_QUEUE_H

/**
 * @brief A very basic thread-safe wrapper for the stl queue (c.f. std::queue)
 * 
 * @tparam T Type to store in queue
 */
template<typename T>
class IQueue
{
public:
	typedef std::shared_ptr<IQueue>  Ptr;

	virtual~IQueue() = default;

	virtual bool empty() const = 0;
	virtual size_t size() = 0;

	virtual T& front() = 0;
	virtual const T& front() const = 0;

	virtual T& back() = 0;

	const T& back() const;

	void push(const T& val);

	void pop();

private:
	mutable std::mutex mtx_;
	std::queue<T> queue_;
};

#endif //I_QUEUE_H
