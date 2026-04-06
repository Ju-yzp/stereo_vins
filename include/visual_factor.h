#ifndef STEREO_VINS_VISUAL_FACTOR_H_
#define STEREO_VINS_VISUAL_FACTOR_H_

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Eigen>

namespace stereo_vins {
// 不对外参进行优化，使用者已经提前使用ka进行外参标定
class VisualFactor : public ceres::SizedCostFunction<2, 6, 6, 1> {
public:
    VisualFactor(
        const Eigen::Vector3d& pts_i_, const Eigen::Vector3d& pts_j_, Eigen::Matrix3d ric,
        Eigen::Vector3d tic);

    virtual bool Evaluate(
        double const* const* parameters, double* residuals, double** jacobians) const;

    // 用于验证计算雅各比矩阵的代码正确性
    // void check(double** parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    Eigen::Matrix2d sqrt_info;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
};
}  // namespace stereo_vins

#endif
