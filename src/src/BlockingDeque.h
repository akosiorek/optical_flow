#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <deque>
#include <mutex>
#include <utility>
#include <condition_variable>

#include "IQueue.h"

/**
 * @brief A very basic thread-safe wrapper for the stl::deque
 * 
 * @tparam T Type to store in queue
 */
template<typename T>
class BlockingDeque : public IQueue<T, std::deque<T> >
{
public:
	typedef std::shared_ptr<BlockingDeque>  Ptr;

	BlockingDeque() {}

	// Element Access
	virtual T& front()
	{
		std::lock_guard<std::mutex> lock(mtx_);
		return deque_.front();
	}

	virtual const T& front() const
	{
		std::lock_guard<std::mutex> lock(mtx_);
		return deque_.front();
	}

	virtual T& back()
	{
		std::lock_guard<std::mutex> lock(mtx_);
		return deque_.back();
	}

	virtual const T& back() const
	{
		std::lock_guard<std::mutex> lock(mtx_);
		return deque_.back();
	}

	// Modifiers
	virtual void push(const T& val)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		deque_.push_back(val);
		lock.unlock();
		cv_.notify_one();
	}

	virtual void pushVector(const std::vector<T>& vals)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		for(auto val : vals)
		{
			deque_.push_back(val);
		}
		lock.unlock();
		cv_.notify_one();
	}

	template<class... Args>
	void emplace(Args&&... args)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		deque_.emplace_back(std::forward<Args>(args)...);
		lock.unlock();
		cv_.notify_one();
	}

	virtual void pop()
	{
		std::unique_lock<std::mutex> lock(mtx_);
        while(deque_.empty())
        {
            cv_.wait(lock);
        }
		deque_.pop_front();
	}

	virtual void swap(std::deque<T>& x)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		deque_.swap(x);
        lock.unlock();
        cv_.notify_one();
	}

	virtual void clear()
	{
		std::lock_guard<std::mutex> lock(mtx_);
		deque_.clear();
	}

	// Combined Modifiers (less locks)
	virtual void pop(T& val)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		while(deque_.empty())
		{
			cv_.wait(lock);
		}
		val = deque_.front();
		deque_.pop_front();
	}

	virtual void pop(std::vector<T>& vals)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		while(deque_.empty())
		{
			cv_.wait(lock);
		}

		if(vals.size()+deque_.size() > vals.capacity())
			vals.reserve(vals.size()+deque_.size());

		for(auto element : deque_)
		{
			vals.push_back(element);
		}

		deque_.clear();
	}

	// Capacity
	virtual bool empty() const
	{
		std::lock_guard<std::mutex> lock(mtx_);
		return deque_.empty();
	}

	virtual std::size_t size() const
	{
		std::lock_guard<std::mutex> lock(mtx_);
		return deque_.size();
	}


private:
	std::deque<T> deque_;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
};

#endif //BLOCKING_QUEUE_H
