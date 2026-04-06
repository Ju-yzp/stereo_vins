#ifndef STEREO_VINS_UTILITY_H_
#define STEREO_VINS_UTILITY_H_

#include <Eigen/Eigen>
#include <iostream>

namespace stereo_vins {

const int MAX_WINDOW_SIZE = 10;

const int MAX_FEATURE_NUM = 1000;

const int TRACK_TIME_THRESHOLD = 4;

const int NUM_ITERATIONS = 20;

const double MIN_DISTANCE = 0.2;
const double MAX_DISTANCE = 10.0;
const double INVALID_DEPTH = -1.0;

struct Observation {
    Observation() : is_stereo(false) {}
    Observation(const Eigen::Matrix<double, 4, 1>& data) : is_stereo(false) {
        uv_raw(0) = data(0);
        uv_raw(1) = data(1);
        uv << data(2), data(3), 1.0;
    }
    void setRightObservation(const Eigen::Matrix<double, 4, 1>& data) {
        uv_r_raw(0) = data(0);
        uv_r_raw(1) = data(1);
        uv_r << data(2), data(3), 1.0;
        is_stereo = true;
    }

    Eigen::Vector2d uv_raw, uv_r_raw;  // 图像原像素
    Eigen::Vector3d uv, uv_r;          // 归一化后的坐标
    bool is_stereo;
};

using FeatureFrame = std::map<int, Observation>;
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d res;
    res << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return res;
}

inline Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi) {
    double theta = phi.norm();
    if (theta < 1e-6) {
        std::cout << "So small number" << std::endl;
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d unit_phi = phi / theta;
    Eigen::Matrix3d R = Eigen::AngleAxisd(theta, unit_phi).toRotationMatrix();

    return R;
}

inline Eigen::Vector3d log(const Eigen::Matrix3d R) {
    double theta = std::acos((R.trace() - 1.0) / 2.0);
    Eigen::Vector3d n;
    if (theta < 1e-6) {
        n << 0, 0, 0;
    } else {
        n << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
        n = n / (2.0 * std::sin(theta));
    }
    Eigen::Vector3d so3 = theta * n;
    return so3;
}
}  // namespace stereo_vins
#endif  // STEREO_VINS_UTILITY_H_
