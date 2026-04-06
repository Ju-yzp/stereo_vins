#include <utility.h>
#include <visual_factor.h>
#include <Eigen/Core>

namespace stereo_vins {

VisualFactor::VisualFactor(
    const Eigen::Vector3d& pts_i_, const Eigen::Vector3d& pts_j_, Eigen::Matrix3d ric_,
    Eigen::Vector3d tic_)
    : pts_i(pts_i_), pts_j(pts_j_), ric(ric_), tic(tic_) {
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp) tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
    sqrt_info = Eigen::Matrix2d::Identity() * 0.02;
}

bool VisualFactor::Evaluate(
    double const* const* parameters, double* residuals, double** jacobians) const {
    Eigen::Vector3d P_i(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Matrix3d R_i =
        expSO3(Eigen::Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));

    Eigen::Vector3d P_j(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Matrix3d R_j =
        expSO3(Eigen::Vector3d(parameters[1][3], parameters[1][4], parameters[1][5]));

    double inv_dep_i = parameters[2][0];
    Eigen::Vector3d pt_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pt_imu_i = ric * pt_camera_i + tic;
    Eigen::Vector3d pt_world_i = R_i * pt_imu_i + P_i;
    Eigen::Vector3d pt_imu_j = R_j.transpose() * (pt_world_i - P_j);
    Eigen::Vector3d pt_camera_j = ric.transpose() * (pt_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = sqrt_info * tangent_base * (pt_camera_j.normalized() - pts_j.normalized());

    if (jacobians) {
        if (pt_camera_j.z() < 0.05) {
            residual.setZero();
            if (jacobians[0])
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jacobians[0]).setZero();
            if (jacobians[1])
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jacobians[1]).setZero();
            if (jacobians[2]) Eigen::Map<Eigen::Vector2d>(jacobians[2]).setZero();
            return true;
        }
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        double norm = pt_camera_j.norm();
        Eigen::Matrix3d jaco_norm =
            1.0 / norm *
            (Eigen::Matrix3d::Identity() - (pt_camera_j * pt_camera_j.transpose()) / (norm * norm));
        reduce = sqrt_info * tangent_base * jaco_norm;
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.block(0, 0, 3, 3) = ric.transpose() * R_j.transpose();
            jaco_i.block(0, 3, 3, 3) =
                -ric.transpose() * R_j.transpose() * R_i * skewSymmetric(pt_imu_i);
            jacobian_pose_i = reduce * jaco_i;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.block(0, 0, 3, 3) = -ric.transpose() * R_j.transpose();
            jaco_j.block(0, 3, 3, 3) = ric.transpose() * skewSymmetric(pt_imu_j);
            jacobian_pose_j = reduce * jaco_j;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[2]);
            jacobian_td = -reduce * ric.transpose() * R_j.transpose() * R_i * ric *
                          (pts_i / (inv_dep_i * inv_dep_i));
        }
    }

    return true;
}

// void VisualFactor::check(double** parameters) {
//     double* res = new double[2];
//     double** jaco = new double*[3];
//     jaco[0] = new double[2 * 7];
//     jaco[1] = new double[2 * 7];
//     jaco[2] = new double[2 * 1];

//     Evaluate(parameters, res, jaco);
//     std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl <<
//     std::endl; std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7,
//     Eigen::RowMajor>>(jaco[0]).block(0, 0, 2, 6)
//               << std::endl
//               << std::endl;
//     std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]).block(0, 0, 2,
//     6)
//               << std::endl
//               << std::endl;
//     std::cout << Eigen::Map<Eigen::Vector2d>(jaco[2]) << std::endl << std::endl;

//     Eigen::Matrix3d R_i =
//         expSO3(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
//     Eigen::Matrix3d R_j =
//         expSO3(Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]));
//     Eigen::Vector3d P_i(parameters[1][0], parameters[1][1], parameters[1][2]);
//     Eigen::Vector3d P_j(parameters[3][0], parameters[3][1], parameters[3][2]);
//     Eigen::Matrix3d ric =
//         expSO3(Eigen::Vector3d(parameters[4][0], parameters[4][1], parameters[4][2]));
//     Eigen::Vector3d tic(parameters[5][0], parameters[5][1], parameters[5][2]);

//     Eigen::Vector3d pt_camera_i = pts_i / inv_dep_i;
//     Eigen::Vector3d pt_imu_i = ric * pt_camera_i + tic;
//     Eigen::Vector3d pt_world_i = R_i * pt_imu_i + P_i;
//     Eigen::Vector3d pt_imu_j = R_j.transpose() * (pt_world_i - P_j);
//     Eigen::Vector3d pt_camera_j = ric.transpose() * (pt_imu_j - tic);

//     Eigen::Vector2d residul =
//         sqrt_info * tangent_base * (pt_camera_j.normalized() - pts_j.normalized());
//     const double eps = 1e-6;
//     Eigen::Matrix<double, 2, 13> delta_residuals;

//     for (int i = 0; i < 13; ++i) {
//         int a = i / 3, b = i % 3;
//         Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
//         Eigen::Matrix3d copy_R_i = R_i, copy_R_j = R_j;
//         Eigen::Vector3d copy_P_i = P_i, copy_P_j = P_j;
//         double copy_inv_dep_i = inv_dep_i;
//         if (a == 0) {
//             copy_P_i += delta;
//         } else if (a == 1) {
//             copy_R_i = copy_R_i * expSO3(delta);
//         } else if (a == 2) {
//             copy_P_j += delta;
//         } else if (a == 3) {
//             copy_R_j = copy_R_j * expSO3(delta);
//         } else {
//             copy_inv_dep_i += delta(0);
//         }

//         pt_camera_i = pts_i / copy_inv_dep_i;
//         pt_imu_i = ric * pt_camera_i + tic;
//         pt_world_i = copy_R_i * pt_imu_i + copy_P_i;
//         pt_imu_j = copy_R_j.transpose() * (pt_world_i - copy_P_j);
//         pt_camera_j = ric.transpose() * (pt_imu_j - tic);

//         Eigen::Vector2d tmp_residul =
//             sqrt_info * tangent_base * (pt_camera_j.normalized() - pts_j.normalized());
//         delta_residuals.col(i) = (tmp_residul - residul) / eps;
//     }

//     std::cout << "------P_i,R_i:------" << std::endl;
//     std::cout << delta_residuals.block(0, 0, 2, 6) << std::endl;
//     std::cout << "------P_j,R_j:------" << std::endl;
//     std::cout << delta_residuals.block(0, 6, 2, 6) << std::endl;
//     std::cout << "------Inv_dep_i:------" << std::endl;
//     std::cout << delta_residuals.block(0, 12, 2, 1) << std::endl;
// }
}  // namespace stereo_vins
