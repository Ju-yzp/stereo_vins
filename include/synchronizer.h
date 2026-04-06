#ifndef STEREO_VINS_SYNCHRONIZER_H_
#define STEREO_VINS_SYNCHRONIZER_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

namespace stereo_vins {

template <typename T>
concept HasTimestamp = requires(T v) {
    { v.get_timestamp() } -> std::convertible_to<double>;
};

enum class Result : uint8_t { SUCCESS = 0, FULL = 1, OUT_OF_SEQUENCE = 2 };

// This class is NOT thread-safe. External synchronization (e.g., a mutex)
// is required if accessed from multiple threads.
template <HasTimestamp T>
class Buffer {
public:
    explicit Buffer(size_t max_capacity) : max_capacity_(max_capacity) {
        data_buffer_ = std::make_shared<std::vector<T>>();
        data_buffer_->reserve(max_capacity);
    }

    void reset() { data_buffer_->clear(); }

    Result push(T data) {
        if (data_buffer_->size() == max_capacity_) {
            return Result::FULL;
        }
        if (!data_buffer_->empty() &&
            data.get_timestamp() <= data_buffer_->back().get_timestamp()) {
            return Result::OUT_OF_SEQUENCE;
        }
        data_buffer_->emplace_back(data);
        return Result::SUCCESS;
    }

    double get_timestamp() {
        return !data_buffer_->empty() ? data_buffer_->front().get_timestamp() : -1;
    }

    std::shared_ptr<Buffer<T>> clone_data(double target_timestamp) {
        std::shared_ptr<Buffer<T>> clone_buffer;
        if (data_buffer_->empty() || data_buffer_->front().get_timestamp() > target_timestamp) {
            return clone_buffer;
        }
        clone_buffer = std::make_shared<Buffer<T>>(max_capacity_);
        if (data_buffer_->back().get_timestamp() <= target_timestamp) {
            std::swap(clone_buffer->data_buffer_, data_buffer_);
        }
        double dt = (data_buffer_->back().get_timestamp() - data_buffer_->front().get_timestamp()) /
                    static_cast<double>(data_buffer_->size());
        size_t predict_idx =
            static_cast<size_t>((target_timestamp - data_buffer_->front().get_timestamp()) / dt);

        auto it = std::upper_bound(
            data_buffer_->begin() + (predict_idx / 2), data_buffer_->end(), target_timestamp,
            [](double t, const T& a) { return t < a.get_timestamp(); });

        clone_buffer->data_buffer_->assign(
            std::make_move_iterator(data_buffer_->begin()), std::make_move_iterator(it));

        data_buffer_->erase(data_buffer_->begin(), it);
        return clone_buffer;
    }

    const std::shared_ptr<std::vector<T>> get_data() { return data_buffer_; }

public:
    std::shared_ptr<std::vector<T>> data_buffer_;
    size_t max_capacity_;
};

template <typename T>
class BufferHeap {
public:
    void push(std::shared_ptr<Buffer<T>> buffer) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (buffer) {
            buffers_.emplace_back(buffer);
            shiftUp(buffers_.size() - 1);
        }
    }

    std::shared_ptr<Buffer<T>> pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        assert(!buffers_.empty() && "Cannot pop from an empty heap!");
        auto top = buffers_.front();
        if (buffers_.size() > 1) {
            buffers_.pop_back();
            shiftDown(0);
        } else {
            buffers_.pop_back();
        }
        return top;
    }

    bool emtry() {
        std::unique_lock<std::mutex> lock(mtx_);
        return buffers_.empty();
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mtx_);
        return buffers_.size();
    }

    std::shared_ptr<Buffer<T>> top() {
        std::unique_lock<std::mutex> lock(mtx_);
        return buffers_.front();
    }

private:
    void shiftUp(size_t id) {
        while (id > 0) {
            size_t parent = (id - 1) / 2;
            if (buffers_[id]->get_timestamp() >= buffers_[parent]->get_timestamp()) break;
            std::swap(buffers_[id], buffers_[parent]);
            id = parent;
        }
    }

    void shiftDown(size_t id) {
        size_t size = buffers_.size();
        while (true) {
            size_t left = 2 * id + 1;
            size_t right = 2 * id + 2;
            size_t smallest = id;

            if (left < size &&
                buffers_[left]->get_timestamp() < buffers_[smallest]->get_timestamp()) {
                smallest = left;
            }
            if (right < size &&
                buffers_[right]->get_timestamp() < buffers_[smallest]->get_timestamp()) {
                smallest = right;
            }
            if (smallest == id) break;

            std::swap(buffers_[id], buffers_[smallest]);
            id = smallest;
        }
    }
    std::vector<std::shared_ptr<Buffer<T>>> buffers_;
    std::mutex mtx_;
};

struct IMUMeasurement {
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
    double get_timestamp() const { return timestamp; }
};

struct StereoFrame {
    double timestamp;
    cv::Mat left_img, right_img;
    double get_timestamp() const { return timestamp; }
};

template <typename T_img, typename T_imu>
struct DataBagTemplate {
    std::shared_ptr<class Buffer<T_img>> img_buffer;
    std::vector<std::shared_ptr<class Buffer<T_imu>>> imu_segments;
};

using DataBag = DataBagTemplate<StereoFrame, IMUMeasurement>;

class Synchronizer {
public:
    Synchronizer() : terminated_(false), last_imu_ts_(0.0) {
        std::lock_guard<std::mutex> lock(free_mtx_);
        for (int i = 0; i < 40; ++i) {
            free_imu_buffers_.push_back(std::make_shared<Buffer<IMUMeasurement>>(100));
            free_stereo_buffers_.push_back(std::make_shared<Buffer<StereoFrame>>(1));
        }
        sync_thread_ = std::thread(&Synchronizer::syncThreadLoop, this);
    }

    ~Synchronizer() {
        terminated_.store(true);
        final_data_cv_.notify_all();
        if (sync_thread_.joinable()) sync_thread_.join();
    }

    void inputStereoFrame(StereoFrame&& frame) {
        auto buf = getFreeStereoBuffer();
        buf->push(std::move(frame));
        stereo_buffer_heap_.push(std::move(buf));
    }

    void inputIMUMeasurement(const IMUMeasurement& imu) {
        std::lock_guard<std::mutex> lock(imu_input_mtx_);
        last_imu_ts_.store(imu.timestamp);

        if (!current_imu_ptr_) {
            current_imu_ptr_ = getFreeImuBuffer();
        }

        if (current_imu_ptr_->push(imu) == Result::FULL) {
            imu_buffer_heap_.push(std::move(current_imu_ptr_));
            current_imu_ptr_ = nullptr;
        }
    }

    bool getFinalData(DataBag& bag, bool wait = false) {
        std::unique_lock<std::mutex> lock(final_data_mtx_);
        if (wait) {
            final_data_cv_.wait(lock, [this] { return !final_data_.empty() || terminated_; });
        }
        if (final_data_.empty()) return false;
        bag = std::move(final_data_.front());
        final_data_.pop_front();
        return true;
    }

private:
    void syncThreadLoop() {
        while (!terminated_.load()) {
            if (stereo_buffer_heap_.emtry() || imu_buffer_heap_.emtry()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            auto img_buf = stereo_buffer_heap_.top();
            double t_target = img_buf->get_timestamp();

            if (imu_buffer_heap_.top()->get_timestamp() > t_target) {
                stereo_buffer_heap_.pop();
                recycleStereoBuffer(img_buf);
                continue;
            }
            if (last_imu_ts_.load() < t_target) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            std::vector<std::shared_ptr<Buffer<IMUMeasurement>>> matched_imu;
            bool sync_ok = false;

            while (!imu_buffer_heap_.emtry()) {
                auto imu_buf = imu_buffer_heap_.pop();
                auto clipped = imu_buf->clone_data(t_target);

                if (clipped) matched_imu.push_back(std::move(clipped));

                if (!imu_buf->data_buffer_->empty()) {
                    imu_buffer_heap_.push(std::move(imu_buf));
                    sync_ok = true;
                    break;
                } else {
                    recycleImuBuffer(std::move(imu_buf));
                }
            }

            if (sync_ok) {
                {
                    std::lock_guard<std::mutex> lock(final_data_mtx_);
                    std::cout << "already packget" << std::endl;
                    DataBag bag;
                    bag.img_buffer = std::move(img_buf);
                    bag.imu_segments = std::move(matched_imu);
                    final_data_.push_back(std::move(bag));
                    final_data_cv_.notify_one();
                }
                stereo_buffer_heap_.pop();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (stereo_buffer_heap_.size() > 50) {
                recycleStereoBuffer(stereo_buffer_heap_.pop());
            }
        }
    }

    std::shared_ptr<Buffer<IMUMeasurement>> getFreeImuBuffer() {
        std::lock_guard<std::mutex> lock(free_mtx_);
        if (free_imu_buffers_.empty()) return std::make_shared<Buffer<IMUMeasurement>>(100);
        auto b = std::move(free_imu_buffers_.back());
        free_imu_buffers_.pop_back();
        return b;
    }

    std::shared_ptr<Buffer<StereoFrame>> getFreeStereoBuffer() {
        std::lock_guard<std::mutex> lock(free_mtx_);
        if (free_stereo_buffers_.empty()) return std::make_shared<Buffer<StereoFrame>>(1);
        auto b = std::move(free_stereo_buffers_.back());
        free_stereo_buffers_.pop_back();
        return b;
    }

    void recycleImuBuffer(std::shared_ptr<Buffer<IMUMeasurement>> b) {
        if (!b) return;
        b->reset();
        std::lock_guard<std::mutex> lock(free_mtx_);
        free_imu_buffers_.push_back(std::move(b));
    }

    void recycleStereoBuffer(std::shared_ptr<Buffer<StereoFrame>> b) {
        if (!b) return;
        b->reset();
        std::lock_guard<std::mutex> lock(free_mtx_);
        free_stereo_buffers_.push_back(std::move(b));
    }

    std::atomic<bool> terminated_;
    std::thread sync_thread_;
    std::atomic<double> last_imu_ts_;

    BufferHeap<IMUMeasurement> imu_buffer_heap_;
    BufferHeap<StereoFrame> stereo_buffer_heap_;

    std::shared_ptr<Buffer<IMUMeasurement>> current_imu_ptr_ = nullptr;
    std::mutex imu_input_mtx_;

    std::mutex free_mtx_;
    std::vector<std::shared_ptr<Buffer<IMUMeasurement>>> free_imu_buffers_;
    std::vector<std::shared_ptr<Buffer<StereoFrame>>> free_stereo_buffers_;

    std::deque<DataBag> final_data_;
    std::mutex final_data_mtx_;
    std::condition_variable final_data_cv_;
};
}  // namespace stereo_vins

#endif
