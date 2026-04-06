#include <featureTracker.h>
#include <utility.h>

// cpp
#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace stereo_vins {
FeatureTracker::FeatureTracker(Params params) {
    rows_ = params.rows;
    cols_ = params.cols;
    grid_rows_ = params.grid_rows;
    grid_cols_ = params.grid_cols;
    border_size_ = params.border_size;
    max_features_ = params.max_features;
    dist_threshold_ = params.dist_threshold;
    mask_radius_ = params.mask_radius;
    flow_back_ = params.flow_back;
    max_move_dist_ = params.max_move_dist;
    enable_gray_gradient_ = params.enable_gray_gradient;
    min_gradient_ = params.min_gradient;
    enable_statistics_ = params.enable_statistics;
    Eigen::Matrix4d Tic = params.T_cam_imu_map[0];
    Eigen::Matrix4d Tic1 = params.T_cam_imu_map[1];
    ric_ = Tic.block<3, 3>(0, 0);
    tic_ = Tic.block<3, 1>(0, 3);
    ric1_ = Tic1.block<3, 3>(0, 0);
    tic1_ = Tic1.block<3, 1>(0, 3);
    for (size_t i = 0; i < 2; ++i) {
        if (params.cam_distort_map.find(i) == params.cam_distort_map.end() ||
            params.cam_intrinsic_map.find(i) == params.cam_intrinsic_map.end()) {
            throw std::runtime_error("Camera parameters missing in Params.");
        }
        k_map_[i] = params.cam_intrinsic_map[i];
        distort_map_[i] = params.cam_distort_map[i];
    }
    grid_x_ = (cols_ - 2 * border_size_) / grid_cols_;
    grid_y_ = (rows_ - 2 * border_size_) / grid_rows_;
    occupy_mask_.resize(grid_x_ * grid_y_);
}

template <typename T>
void remove_by_indices(std::vector<T>& vec, std::vector<int>& remove_indices) {
    std::sort(remove_indices.rbegin(), remove_indices.rend());

    for (int idx : remove_indices) {
        if (idx < vec.size()) {
            vec.erase(vec.begin() + idx);
        }
    }
}

FeatureFrame FeatureTracker::feedStereo(const cv::Mat& left_img, const cv::Mat& right_img) {
    cur_img_ = left_img;
    cur_ids_ = prev_ids_;
    if (!prev_pts_.empty()) {
        monocularMatching(prev_img_, cur_img_, prev_pts_, cur_pts_, prev_ids_, cur_ids_);
    }

    if (static_cast<int>(cur_pts_.size()) < max_features_ * 0.7) {
        setMask();
        std::vector<cv::Point2f> new_pts;
        std::vector<cv::Point2f> grad;
        replenishKeypoints(exist_mask_, new_pts, grad);
        for (size_t i = 0; i < new_pts.size(); ++i) {
            cur_pts_.emplace_back(new_pts[i]);
            ++feature_id_;
            cur_ids_.emplace_back(feature_id_);
            if (enable_gray_gradient_) {
                gray_gradient_map_[feature_id_] = grad[i];
            }
            if (enable_statistics_) {
                tracked_times_[feature_id_] = 0;
            }
        }
    }

    // 统计追踪特征点次数
    if (enable_statistics_) {
        for (size_t id : cur_ids_) {
            ++tracked_times_[id];
        }
    }

    // 删除失去追踪的特征点的统计和梯度信息,避免开启debug选项时导致使用内存持续增长
    std::map<size_t, size_t> new_tracked_times;
    std::map<size_t, cv::Point2f> new_gray_gradient_map;
    for (size_t id : cur_ids_) {
        new_tracked_times[id] = tracked_times_[id];
        if (enable_gray_gradient_) {
            new_gray_gradient_map[id] = gray_gradient_map_[id];
        }
    }
    tracked_times_ = std::move(new_tracked_times);
    gray_gradient_map_ = std::move(new_gray_gradient_map);

    right_ids_.clear();
    cur_rpts_.clear();
    right_ids_ = cur_ids_;
    monocularMatching(cur_img_, right_img, cur_pts_, cur_rpts_, cur_ids_, right_ids_);

    FeatureFrame feature_frame;
    std::set<size_t> removed_bad_indices;
    for (size_t i = 0; i < cur_ids_.size(); ++i) {
        size_t id = cur_ids_[i];
        cv::Point2f uv_raw = cur_pts_[i];

        auto uv = undistort(0, uv_raw);

        if (uv.has_value() && std::abs(uv->x) < 2.0 && std::abs(uv->y) < 2.0) {
            Eigen::Matrix<double, 4, 1> data;
            // std::cout<<uv.value().x<<" "<<uv.value().y<<std::endl;
            data << uv_raw.x, uv_raw.y, uv.value().x, uv.value().y;
            feature_frame[id] = Observation(data);
        } else {
            removed_bad_indices.insert(i);
        }
    }

    size_t slow = 0;
    for (size_t fast = 0; fast < cur_ids_.size(); ++fast) {
        if (removed_bad_indices.find(fast) == removed_bad_indices.end()) {
            if (slow != fast) {
                cur_ids_[slow] = cur_ids_[fast];
                cur_pts_[slow] = cur_pts_[fast];
            }
            slow++;
        }
    }

    for (size_t i = 0; i < right_ids_.size(); ++i) {
        size_t id = right_ids_[i];
        cv::Point2f uv_raw = cur_rpts_[i];
        auto uv = undistort(1, uv_raw);
        if (uv.has_value() && feature_frame.find(id) != feature_frame.end()) {
            Eigen::Matrix<double, 1, 4> data;
            data << uv_raw.x, uv_raw.y, uv.value().x, uv.value().y;
            feature_frame[id].setRightObservation(data);
            if (isBadStereoPoint(feature_frame[id].uv, feature_frame[id].uv_r)) {
                feature_frame[id].is_stereo = false;
            }
        }
    }

    cur_ids_.resize(slow);
    cur_pts_.resize(slow);
    std::swap(prev_img_, cur_img_);
    std::swap(prev_pts_, cur_pts_);
    std::swap(prev_ids_, cur_ids_);
    cur_ids_.clear();
    cur_pts_.clear();
    cur_img_.release();
    return feature_frame;
}

void FeatureTracker::reset() {
    prev_pts_.clear();
    cur_pts_.clear();
    cur_rpts_.clear();
    prev_ids_.clear();
    cur_ids_.clear();
    tracked_times_.clear();
    gray_gradient_map_.clear();
    exist_mask_.release();
    cur_img_.release();
    prev_img_.release();
    feature_id_ = 0;
}

cv::Mat FeatureTracker::showTrackResult(const cv::Mat& left_img, const cv::Mat& right_img) {
    cv::Mat track_result;
    cv::hconcat(left_img, right_img, track_result);
    cv::cvtColor(track_result, track_result, cv::COLOR_GRAY2BGR);

    if (enable_gray_gradient_) {
        for (size_t i = 0; i < prev_ids_.size(); ++i) {
            cv::Point2f pt = prev_pts_[i];
            cv::Point2f grad = gray_gradient_map_[prev_ids_[i]];
            Eigen::Vector2d direction = Eigen::Vector2d(grad.x, grad.y).normalized();
            cv::Point2f end_pt = pt + cv::Point2f(direction.x(), direction.y()) * 6.0f;
            cv::arrowedLine(
                track_result, pt, end_pt, cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.3);
        }
    }
    if (enable_statistics_) {
        for (size_t i = 0; i < prev_ids_.size(); ++i) {
            double ratio = std::min(static_cast<double>(tracked_times_[prev_ids_[i]]), 30.0) / 30.0;
            cv::Point2f pt = prev_pts_[i];
            cv::circle(
                track_result, prev_pts_[i], 2, cv::Scalar(255 * (1.0 - ratio), 0, 255 * ratio), 2);
        }
    } else {
        for (auto pt : cur_pts_) {
            cv::circle(track_result, pt, 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    for (auto pt : cur_rpts_) {
        cv::Point2f new_pt(pt.x + cols_, pt.y);
        cv::circle(track_result, new_pt, 2, cv::Scalar(0, 255, 0), -1);
    }
    return track_result;
}

void FeatureTracker::monocularMatching(
    const cv::Mat& prev_img, const cv::Mat& cur_img, std::vector<cv::Point2f>& prev_pts,
    std::vector<cv::Point2f>& cur_pts, std::vector<size_t>& prev_ids,
    std::vector<size_t>& cur_ids) {
    std::vector<uchar> p2c_status;
    std::vector<float> p2c_err;
    size_t num = prev_pts.size();
    cv::calcOpticalFlowPyrLK(
        prev_img, cur_img, prev_pts, cur_pts, p2c_status, p2c_err, cv::Size(21, 21), 3);

    if (flow_back_) {
        std::vector<cv::Point2f> copy_pts = prev_pts;
        std::vector<uchar> c2p_status;
        std::vector<float> c2p_err;
        cv::calcOpticalFlowPyrLK(
            cur_img, prev_img, cur_pts, copy_pts, c2p_status, c2p_err, cv::Size(21, 21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        for (int i = 0; i < num; ++i) {
            p2c_status[i] = p2c_status[i]
                                ? (c2p_status[i] &&
                                   (computeMoveDist(prev_pts[i], copy_pts[i]) < max_move_dist_))
                                : 0;
        }
    }

    size_t valid_count = 0;
    for (size_t index = 0; index < num; ++index) {
        if (p2c_status[index] && isInsideImage(cur_pts[index])) {
            cur_pts[valid_count] = cur_pts[index];
            cur_ids[valid_count] = prev_ids[index];
            ++valid_count;
        }
    }

    cur_pts.resize(valid_count);
    cur_ids.resize(valid_count);
}

void FeatureTracker::setMask() {
    exist_mask_ = cv::Mat(rows_, cols_, CV_8UC1, cv::Scalar(255));
    occupy_mask_.assign(occupy_mask_.size(), false);
    for (auto& center : cur_pts_) {
        cv::circle(exist_mask_, center, mask_radius_, cv::Scalar(0), -1);
        int y = center.y - border_size_;
        int x = center.x - border_size_;
        if (y < rows_ - border_size_ && x < cols_ - border_size_) {
            int id = x / grid_cols_ + (y / grid_rows_) * grid_x_;
            if (id < static_cast<int>(occupy_mask_.size()) && id >= 0) {
                occupy_mask_[id] = true;
            }
        }
    }
}

void FeatureTracker::replenishKeypoints(
    const cv::Mat& mask, std::vector<cv::Point2f>& new_pts, std::vector<cv::Point2f>& grad) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(cur_img_, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(cur_img_, grad_y, CV_32F, 0, 1, 3);

    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(grad_x, grad_x, Ix2);
    cv::multiply(grad_y, grad_y, Iy2);
    cv::multiply(grad_x, grad_y, Ixy);

    int block_size = 3;
    cv::boxFilter(Ix2, Ix2, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(Iy2, Iy2, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(Ixy, Ixy, CV_32F, cv::Size(block_size, block_size));

    cv::Mat score_map = cv::Mat::zeros(cur_img_.size(), CV_32F);
    float* Ix2_ptr = reinterpret_cast<float*>(Ix2.data);
    float* Iy2_ptr = reinterpret_cast<float*>(Iy2.data);
    float* Ixy_ptr = reinterpret_cast<float*>(Ixy.data);
    for (int y = 0; y < rows_; ++y) {
        for (int x = 0; x < cols_; ++x) {
            size_t offset = y * cols_ + x;
            float a = Ix2_ptr[offset];
            float b = Ixy_ptr[offset];
            float c = Iy2_ptr[offset];
            float term = std::sqrt((a - c) * (a - c) + 4 * b * b);
            score_map.at<float>(y, x) = (a + c - term) * 0.5f;
        }
    }

    float* score_ptr = reinterpret_cast<float*>(score_map.data);
    float* grad_x_ptr = reinterpret_cast<float*>(grad_x.data);
    float* grad_y_ptr = reinterpret_cast<float*>(grad_y.data);
    for (int grid_y = border_size_, row = 0; grid_y < rows_ - border_size_;
         grid_y += grid_rows_, ++row) {
        for (int grid_x = border_size_, col = 0; grid_x < cols_ - border_size_;
             grid_x += grid_cols_, ++col) {
            if (occupy_mask_[row * grid_x + col]) {
                continue;
            }
            float best_score = min_gradient_;
            cv::Point best_pt(-1, -1);

            for (int y = grid_y; y < std::min(grid_y + grid_rows_, rows_); ++y) {
                for (int x = grid_x; x < std::min(grid_x + grid_cols_, cols_); ++x) {
                    float s = score_ptr[y * cols_ + x];
                    if (s > best_score && mask.at<uchar>(y, x) > 0) {
                        best_score = s;
                        best_pt = cv::Point(x, y);
                    }
                }
            }

            if (best_pt.x != -1) {
                new_pts.emplace_back(best_pt);
                grad.emplace_back(cv::Point2f(
                    grad_x_ptr[best_pt.y * cols_ + best_pt.x],
                    grad_y_ptr[best_pt.y * cols_ + best_pt.x]));
            }
        }
    }
}

std::optional<cv::Point2f> FeatureTracker::undistort(size_t id, const cv::Point2f& uv_raw) {
    std::vector<cv::Point2f> dist_pts = {uv_raw};
    std::vector<cv::Point2f> undist_pts;
    cv::undistortPoints(dist_pts, undist_pts, k_map_[id], distort_map_[id]);
    cv::Point2f pt = undist_pts[0];

    const double NORMALIZED_BOUND = 2.0;
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || std::abs(pt.x) > NORMALIZED_BOUND ||
        std::abs(pt.y) > NORMALIZED_BOUND) {
        return std::nullopt;
    }

    return pt;
}

bool FeatureTracker::isBadStereoPoint(Eigen::Vector3d uv, Eigen::Vector3d uv_r) {
    Eigen::Matrix<double, 3, 4> pose0 = Eigen::Matrix<double, 3, 4>::Identity();
    Eigen::Matrix<double, 3, 4> pose1;
    pose1.block<3, 3>(0, 0) = ric1_.transpose() * ric_;
    pose1.block<3, 1>(0, 3) = ric1_.transpose() * (tic_ - tic1_);
    Eigen::Matrix4d A;
    A.row(0) = uv.x() * pose0.row(2) - pose0.row(0);
    A.row(1) = uv.y() * pose0.row(2) - pose0.row(1);
    A.row(2) = uv_r.x() * pose1.row(2) - pose1.row(0);
    A.row(3) = uv_r.y() * pose1.row(2) - pose1.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);
    Eigen::Vector3d point_3d = homogeneous_point.head<3>() / homogeneous_point.w();
    if (point_3d.z() > MIN_DISTANCE && point_3d.z() < MAX_DISTANCE) {
        Eigen::Vector3d new_pt = pose1.block<3, 3>(0, 0) * point_3d + pose1.block<3, 1>(0, 3);
        if (new_pt.z() > MIN_DISTANCE && new_pt.z() < 3.0) {
            return false;
        }
    }

    return true;
}
}  // namespace stereo_vins
