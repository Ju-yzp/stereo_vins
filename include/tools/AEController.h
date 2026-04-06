/**
 * @file AEController.h
 * @brief Smooth Exposure-Gain Trajectory Planning for VINS.
 * * This module implements a 2D B-spline based auto-exposure controller.
 * It manages the trade-off between Exposure Time and Sensor Gain to optimize
 * photometric consistency for KLT feature tracking.
 * * @note Core B-spline trajectory logic is inspired by and adapted from:
 * Fast-Planner (HKUST Aerial Robotics Group).
 * Repository: https://github.com/HKUST-Aerial-Robotics/Fast-Planner
 * * @authors
 * - JunPing Wu (Implementation for stereo_vins)
 * - Boyu Zhou (Original B-spline optimization logic in Fast-Planner)
 * * @license
 * This project incorporates algorithmic principles from Fast-Planner.
 * Please refer to the original Fast-Planner license (LGPLv3) when
 * redistributing the B-spline core components.
 */

#ifndef STEREO_VINS_AUTO_EXPOSURE_CONTROLLER_H_
#define STEREO_VINS_AUTO_EXPOSURE_CONTROLLER_H_

// cpp
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// eigen
#include <Eigen/Core>

namespace stereo_vins {

/**
 * @brief Generic camera control callback.
 * * Allows the registration of hardware-specific control interfaces, making
 * the AE algorithm "camera-type agnostic" (decoupled from DepthAI/ROS/etc).
 * @param exposure_us The calculated target exposure time in microseconds.
 */
using ControlCallback = std::function<void(double)>;

class AutoExposureController {
public:
    AutoExposureController(const ControlCallback& control_cb)
        : p_(3),
          n_(3),
          m_(p_ + n_ + 1),
          control_cb_(control_cb),
          prev_timestamp_(-1.0),
          tolerance_luma_(15.0),
          max_exposuretime_(20000.0),
          limit_intensity_vel_(0.025),
          limit_intensity_acc_(0.1) {
        u_.resize(m_ + 1, 0.0);
    }

    void changeExposureTime(
        const cv::Mat& origin_gray, double timestamp, double actual_exposure,
        std::string camera_name) {
        double total_gray = 0;
        for (int y = 0; y < origin_gray.rows; ++y) {
            for (int x = 0; x < origin_gray.cols; ++x) {
                total_gray += origin_gray.at<uchar>(y, x);
            }
        }
        double cur_luma = total_gray / static_cast<double>(origin_gray.rows * origin_gray.cols);

        std::cout << camera_name << " " << cur_luma << std::endl;
        std::cout << camera_name << "actually exposure time" << actual_exposure << std::endl;
        double cur_light_intensity = cur_luma / std::max(actual_exposure, 1.0);

        if (std::abs(target_luma_ - cur_luma) < tolerance_luma_) {
            control_points_.emplace_back(Eigen::Vector2d(timestamp, cur_light_intensity));

            if (control_points_.size() > 2) {
                control_points_.erase(control_points_.begin(), control_points_.end() - 2);
            }

            prev_timestamp_ = timestamp;
            return;
        }

        double expected_exposuretime = actual_exposure * (target_luma_ / cur_luma);
        expected_exposuretime = std::min(std::max(expected_exposuretime, 100.0), max_exposuretime_);
        double target_light_intensity = target_luma_ / expected_exposuretime;

        control_points_.emplace_back(Eigen::Vector2d(timestamp, cur_light_intensity));

        if (control_points_.size() == 3) {
            double delta_l = std::abs(target_light_intensity - control_points_.back()(1));

            double dt = std::max((delta_l / limit_intensity_vel_) * 1.3, 0.033);
            double target_timestamp = control_points_.back()(0) + dt;

            control_points_.emplace_back(Eigen::Vector2d(target_timestamp, target_light_intensity));

            recomputeKnots();

            if (!checkFeasibility()) {
                target_timestamp += dt * 0.05;
                control_points_.back()(0) = target_timestamp;
                recomputeKnots();
            }

            double eval_time = timestamp + 0.033;
            double predicted_light = evaluateDeBoor(eval_time);

            double predict_exposuretime = target_luma_ / std::max(predicted_light, 1e-6);
            predict_exposuretime =
                std::min(std::max(predict_exposuretime, 100.0), max_exposuretime_);

            control_cb_(predict_exposuretime);

            control_points_.pop_back();
            control_points_.erase(control_points_.begin());

            prev_exposuretime_ = predict_exposuretime;
            prev_timestamp_ = timestamp;
            prev_luma_ = cur_luma;
        }
    }

    void setTargetLuma(double target_luma) { target_luma_ = target_luma; }

    void setToleranceLuma(double tolerance_luma) { tolerance_luma_ = tolerance_luma; }

private:
    void recomputeKnots() {
        assert(control_points_.size() == p_ + 1 && "Control points must match spline order + 1");
        double dt[3];
        for (int i = 1; i < 4; ++i) {
            dt[i - 1] = control_points_[i](0) - control_points_[i - 1](0);
            if (dt[i - 1] < 1e-4) dt[i - 1] = 1e-4;
        }
        for (int i = 0; i <= m_; ++i) {
            if (i <= p_) {
                u_[i] = control_points_[0](0) + double(-p_ + i) * dt[0];
            } else if (i > p_ && i <= m_ - p_) {
                u_[i] = u_[i - 1] + dt[1];
            } else if (i > m_ - p_) {
                u_[i] = u_[i - 1] + dt[2];
            }
        }
    }

    double evaluateDeBoor(const double& u) {
        double ub = std::min(std::max(u_[p_], u), u_[m_ - p_]);
        int k = p_;
        while (k < m_ - p_) {
            if (u_[k + 1] > ub) break;
            ++k;
        }

        std::vector<double> d;
        for (int i = 0; i <= p_; ++i) {
            d.push_back(control_points_[k - p_ + i](1));
        }

        for (int r = 1; r <= p_; ++r) {
            for (int i = p_; i >= r; --i) {
                double denominator = u_[i + 1 + k - r] - u_[i + k - p_];
                if (denominator < 1e-6) denominator = 1e-6;
                double alpha = (ub - u_[i + k - p_]) / denominator;
                d[i] = (1 - alpha) * d[i - 1] + alpha * d[i];
            }
        }
        return d[p_];
    }

    bool checkFeasibility() {
        bool fea = true;
        int size = static_cast<int>(control_points_.size());

        for (int i = 0; i < size - 1; ++i) {
            double dt = u_[i + p_ + 1] - u_[i + 1];
            if (dt < 1e-6) continue;
            double vel = p_ * (control_points_[i + 1](1) - control_points_[i](1)) / dt;
            if (std::abs(vel) > limit_intensity_vel_ + 1e-4) fea = false;
        }

        for (int i = 0; i < size - 2; ++i) {
            double dt1 = u_[i + p_ + 1] - u_[i + 1];
            double dt2 = u_[i + p_ + 2] - u_[i + 2];
            double dt_acc = u_[i + p_ + 1] - u_[i + 2];

            if (dt1 < 1e-6 || dt2 < 1e-6 || dt_acc < 1e-6) continue;

            double vel1 = (control_points_[i + 1](1) - control_points_[i](1)) / dt1;
            double vel2 = (control_points_[i + 2](1) - control_points_[i + 1](1)) / dt2;
            double acc = p_ * (p_ - 1) * (vel2 - vel1) / dt_acc;

            if (std::abs(acc) > limit_intensity_acc_ + 1e-4) fea = false;
        }
        return fea;
    }

    int p_, n_, m_;
    double max_exposuretime_;
    double limit_intensity_vel_;
    double limit_intensity_acc_;
    std::vector<Eigen::Vector2d> control_points_;
    std::vector<double> u_;
    double target_luma_;
    double tolerance_luma_;
    ControlCallback control_cb_;
    double prev_timestamp_;
    double prev_exposuretime_;
    double prev_luma_;
};
}  // namespace stereo_vins

#endif
