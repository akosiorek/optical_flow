#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <queue>
#include <mutex>

/**
 * @brief A very basic thread-safe wrapper for the stl queue (c.f. std::queue)
 * 
 * @tparam T Type to store in queue
 */
template<typename T>
class BlockingQueue
{
public:
	typedef  std::shared_ptr<BlockingQueue>  Ptr;

	BlockingQueue() {}

	bool empty() const
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.empty();
	}

	size_t size()
	{
		std::unique_lock<std::mutex> lck (mtx_);
		return queue_.size();
	}	

	T& front()
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.front();
	}

	const T& front() const
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.front();
	}

	T& back()
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.back();
	}

	const T& back() const
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.back();
	}

	//NOT TESTED
	void emplace(T&& val)
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.emplace(val);
	}

	void push(const T& val)
	{
		std::unique_lock<std::mutex> lck (mtx_);
		queue_.push(val);
	}

	void pop()
	{
		std::unique_lock<std::mutex> lck (mtx_);		
		return queue_.pop();
	}


private:
	std::mutex		mtx_;	
	std::queue<T> 	queue_;
};

#endif //BLOCKING_QUEUE_H
