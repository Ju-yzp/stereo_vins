#ifndef STEREO_VINS_PREINTEGRATOR_H_
#define STEREO_VINS_PREINTEGRATOR_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>

namespace stereo_vins {

class Preintegrator {
public:
    Preintegrator();

    void addMeasurement(
        const Eigen::Vector3d& acceleration, const Eigen::Vector3d& gryroscope, double delta_time);

    void clearMeasurements();

    void midpointIntegration(
        double dt, Eigen::Vector3d acc_0, Eigen::Vector3d gyr_0, Eigen::Vector3d acc_1,
        Eigen::Vector3d gyr_1, Eigen::Matrix3d& result_rotation, Eigen::Vector3d& result_position,
        Eigen::Vector3d& result_vec) {
        // 中值积分过程
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - bias_gryroscope_;
        result_rotation = delta_rotation_ *
                          Eigen::AngleAxisd(un_gyr.norm(), un_gyr.normalized()).toRotationMatrix();
        Eigen::Vector3d un_acc = delta_rotation_ * ((acc_1 + acc_0) * 0.5 - bias_acceleration_);
        result_position = delta_position_ + delta_velocity_ * dt + 0.5 * un_acc * dt * dt;
        result_vec = delta_velocity_ + delta_velocity_ * dt;

        // 更新雅各比和协方差矩阵
        Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 18>::Zero();
        Eigen::Vector3d acc0 = acc_0 - bias_acceleration_;
        Eigen::Vector3d acc1 = acc_1 - bias_acceleration_;

        F.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
        F.block(0, 3, 3, 3) = -0.25 * delta_rotation_ * skewSymmetric(acc0) -
                              0.25 * result_rotation * skewSymmetric(acc1) *
                                  (Eigen::Matrix3d::Identity() *
                                   skewSymmetric((gyr_0 + gyr_1) * 0.5 - bias_gryroscope_));
        F.block(0, 6, 3, 3) = Eigen::Matrix3d::Identity() * dt;
        F.block(0, 9, 3, 3) = -0.25 * (result_rotation + delta_rotation_) * dt;
        F.block(0, 12, 3, 3) = 0.25 * result_rotation * skewSymmetric(acc1) * dt * dt * dt;

        F.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity() -
                              skewSymmetric(0.5 * (gyr_0 + gyr_1) - bias_gryroscope_) * dt;
        F.block(3, 12, 3, 3) = -Eigen::Matrix3d::Identity() * dt;

        F.block(6, 3, 3, 3) = -0.5 * delta_rotation_ * skewSymmetric(acc0) * dt -
                              0.5 * result_rotation * skewSymmetric(acc1) *
                                  (Eigen::Matrix3d::Identity() -
                                   skewSymmetric(0.5 * (gyr_0 + gyr_1) - bias_gryroscope_) * dt) *
                                  dt;

        F.block(6, 6, 3, 3) = Eigen::Matrix3d::Identity();
        F.block(6, 9, 3, 3) = -0.5 * (delta_rotation_ + result_rotation) * dt;
        F.block(6, 12, 3, 3) = 0.5 * result_rotation * skewSymmetric(acc1) * dt * dt;

        F.block(9, 9, 3, 3) = Eigen::Matrix3d::Identity();
        F.block(12, 12, 3, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 15, 18> V = Eigen::Matrix<double, 15, 18>::Zero();
        V.block(0, 0, 3, 3) = V.block(0, 9, 3, 3) = -0.25 * delta_rotation_ * dt * dt;
        V.block(0, 3, 3, 3) = 0.125 * result_rotation * skewSymmetric(acc1) * dt * dt * dt;
        V.block(0, 6, 3, 3) = -0.25 * result_rotation * dt * dt;

        V.block(3, 3, 3, 3) = V.block(3, 9, 3, 3) = -Eigen::Matrix3d::Identity() * 0.5 * dt;

        V.block(6, 0, 3, 3) = -0.5 * dt * delta_rotation_;
        V.block(6, 9, 3, 3) = V.block(6, 3, 3, 3) = 0.5 * V.block(0, 0, 3, 3);
        V.block(6, 6, 3, 3) = -0.5 * dt * result_rotation;
    }

    void propagate(
        double dt, const Eigen::Vector3d& acc_0, const Eigen::Vector3d& acc_1,
        const Eigen::Vector3d& gyr_0, const Eigen::Vector3d& gyr_1) {}

private:
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d res;
        res << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return res;
    }

    Eigen::Matrix3d computeRightJacobianSO3(const Eigen::Vector3d& phi) {
        double theta = phi.norm();
        Eigen::Matrix3d phi_hat;
        phi_hat << 0, -phi(2), phi(1), phi(2), 0, -phi(0), -phi(1), phi(0), 0;

        if (theta < 1e-6) {
            return Eigen::Matrix3d::Identity() - 0.5 * phi_hat + (1.0 / 6.0) * phi_hat * phi_hat;
        } else {
            Eigen::Matrix3d phi_hat_2 = phi_hat * phi_hat;
            return Eigen::Matrix3d::Identity() -
                   ((1.0 - std::cos(theta)) / (theta * theta)) * phi_hat +
                   ((theta - std::sin(theta)) / (theta * theta * theta)) * phi_hat_2;
        }
    }
    // IMU测量数据缓存
    std::vector<Eigen::Vector3d> acceleration_buffer_;
    std::vector<Eigen::Vector3d> gryroscope_buffer_;
    std::vector<double> delta_time_buffer_;

    // 预积分结果
    Eigen::Vector3d delta_position_;
    Eigen::Matrix3d delta_rotation_;
    Eigen::Vector3d delta_velocity_;

    // 偏置
    Eigen::Vector3d bias_acceleration_;
    Eigen::Vector3d bias_gryroscope_;

    // 初始速度
    const Eigen::Vector3d init_acc_;

    // 协方差矩阵
    Eigen::Matrix<double, 18, 18> noise_;
    Eigen::Matrix<double, 15, 15> covariance_;
};
}  // namespace stereo_vins

#endif  // STEREO_VINS_PREINTEGRATOR_H_
