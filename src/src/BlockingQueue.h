#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <queue>
#include <mutex>
#include <thread>

/**
 * @brief A very basic thread-safe wrapper for the stl queue (c.f. std::queue)
 *
 * @tparam T Type to store in queue
 */
template<typename T>
class BlockingQueue
{
public:
	typedef std::shared_ptr<BlockingQueue>  Ptr;

	BlockingQueue() {}

	bool empty() const
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.empty();
	}

	size_t size()
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.size();
	}

	T& front()
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.front();
	}

	const T& front() const
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.front();
	}

	T& back()
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.back();
	}

	const T& back() const
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.back();
	}

	//NOT TESTEDz
	void emplace(T&& val)
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.emplace(val);
	}

	void push(const T& val)
	{
		std::lock_guard<std::mutex> lck (mtx_);
		queue_.push(val);
	}

	void pop()
	{
		std::lock_guard<std::mutex> lck (mtx_);
		return queue_.pop();
	}


private:
	mutable std::mutex mtx_;
	std::queue<T> queue_;
};

#endif //BLOCKING_QUEUE_H
