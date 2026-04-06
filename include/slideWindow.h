#ifndef STEREO_VINS_SLIDE_WINDOW_H_
#define STEREO_VINS_SLIDE_WINDOW_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

namespace stereo_vins {
struct SlideWindow {
    // 信息矩阵，代表这个窗口的位姿置信度
    Eigen::Matrix2d sqrt_info;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
};
}  // namespace stereo_vins

#endif
