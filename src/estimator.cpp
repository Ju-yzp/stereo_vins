#include <ceres/manifold.h>
#include <estimator.h>
#include <slideWindow.h>
#include <utility.h>
#include <visual_factor.h>
#include <Eigen/Core>
#include <cmath>
#include <vector>

namespace stereo_vins {
void StateEstimator::processMeasurement(FeatureFrame& feature_frame) {
    // if (!init_imu_pose_.load()) {
    //     std::vector<Eigen::Vector3d> acc_vec;
    //     for (auto imu_data : measurement.imu_measurements) {
    //         acc_vec.emplace_back(imu_data.acc);
    //     }

    //     initImuPose(acc_vec);
    // } else {
    if (init_imu_pose_.load()) {
        // fast_predict_state(measurement);
        if (feature_manager_->checkKeyFrameByParallax(frame_count_, feature_frame)) {
            if (frame_count_ != MAX_WINDOW_SIZE) {
                feature_manager_->initFramePoseByPNP(frame_count_, windows);

                feature_manager_->initFeaturesDepth(
                    windows,
                    ((windows[frame_count_].P - windows[frame_count_ - 1].P).norm() > 0.05));

                std::set<int> outlier_ids = outliersReprojection();
                feature_manager_->removeOutliers(outlier_ids);
                ++frame_count_;
            } else {
                feature_manager_->initFramePoseByPNP(frame_count_, windows);

                feature_manager_->initFeaturesDepth(
                    windows,
                    ((windows[frame_count_].P - windows[frame_count_ - 1].P).norm() > 0.05));
                // stateToParams();
                // optimize();
                // paramsToState();
                std::cout << "new key frame" << std::endl;
                std::set<int> outlier_ids = outliersReprojection();
                feature_manager_->removeOutliers(outlier_ids);
                slideWindow();
            }
        } else {
            feature_manager_->removeNewestFeatures(frame_count_);
        }
        // if (frame_count_ == MAX_WINDOW_SIZE) {
        //     if (is_keyframe) {
        //         slideWindow();
        //     } else if (!is_keyframe) {
        //         feature_manager_->removeNewestFeatures(frame_count_);
        //     }
        // }
    }
}
// void StateEstimator::fast_predict_state(const Measurement& measurement) {
//     ImuMeasurement prev_meas;
//     for (auto imu_data : measurement.imu_measurements) {
//         double dt = imu_data.ts - prev_meas.ts;
//         Eigen::Vector3d un_gyr = 0.5 * (prev_meas.gryo + imu_data.gryo) - Bg;
//         Eigen::Matrix3d temp_r;
//         temp_r = R * Eigen::AngleAxisd(un_gyr.norm(), un_gyr.normalized()).toRotationMatrix();
//         Eigen::Vector3d un_acc = R * ((imu_data.acc + prev_meas.acc) * 0.5 - Ba);
//         P = P + V * dt + 0.5 * un_acc * dt * dt;
//         V += un_acc * dt;
//     }

//     windows[frame_count_].P = P;
//     windows[frame_count_].R = R;
// }
void StateEstimator::optimize() {
    stateToParams();

    ceres::LossFunction* loss_function;
    loss_function = new ceres::HuberLoss(1.0);

    ceres::Problem problem;
    for (int i = 0; i < frame_count_ + 1; i++) {
        double* current_pose = para_pose_[i];

        auto* translation_manifold = new ceres::EuclideanManifold<3>();

        auto* rotation_vector_manifold = new ceres::EuclideanManifold<3>();

        auto* se3_manifold =
            new ceres::ProductManifold<ceres::EuclideanManifold<3>*, ceres::EuclideanManifold<3>*>(
                new ceres::EuclideanManifold<3>(), new ceres::EuclideanManifold<3>());

        problem.AddParameterBlock(current_pose, 6);
        problem.SetManifold(current_pose, se3_manifold);
    }

    problem.SetParameterBlockConstant(para_pose_[0]);

    feature_num_ = 0;
    for (const auto& [id, feature] : feature_manager_->get_all_features()) {
        if (feature_num_ > MAX_FEATURE_NUM) {
            break;
        }

        const std::vector<Observation>& obs_vec = feature.obs_vec;
        double depth = feature.depth_estimation;
        if (static_cast<int>(obs_vec.size()) < TRACK_TIME_THRESHOLD || depth == INVALID_DEPTH) {
            continue;
        }

        int frame_i = feature.start_frame_id;
        int frame_j = frame_i - 1;
        Eigen::Vector3d pt_i = obs_vec[0].uv;
        double inv_dep_i = 1.0 / feature.depth_estimation;
        for (const auto& obs : obs_vec) {
            ++frame_j;
            if (frame_i != frame_j) {
                Eigen::Vector3d pt_j = obs.uv;
                VisualFactor* visual_factor = new VisualFactor(pt_i, pt_j, ric_, tic_);
                problem.AddResidualBlock(
                    visual_factor, loss_function, para_pose_[frame_i], para_pose_[frame_j],
                    &para_features_[feature_num_]);
            }
        }
        para_features_[feature_num_] = inv_dep_i;
        ++feature_num_;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = 0.05;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    paramsToState();
}

void StateEstimator::stateToParams() {
    for (int id = 0; id < MAX_WINDOW_SIZE; ++id) {
        para_pose_[id][0] = windows[id].P.x();
        para_pose_[id][1] = windows[id].P.y();
        para_pose_[id][2] = windows[id].P.z();

        Eigen::Vector3d q = log(windows[id].R);
        para_pose_[id][3] = q.x();
        para_pose_[id][4] = q.y();
        para_pose_[id][5] = q.z();
    }
}

void StateEstimator::paramsToState() {
    for (int id = 0; id < frame_count_; ++id) {
        windows[id].P.x() = para_pose_[id][0];
        windows[id].P.y() = para_pose_[id][1];
        windows[id].P.z() = para_pose_[id][2];
        Eigen::Vector3d rot_vec(para_pose_[id][3], para_pose_[id][4], para_pose_[id][5]);

        if (rot_vec.norm() > 1e-12) {
            windows[id].R = expSO3(rot_vec);
        } else {
            windows[id].R = Eigen::Matrix3d::Identity();
        }
    }

    int err_num = 0;
    for (auto& [id, feature] : feature_manager_->get_all_features()) {
        if (err_num >= feature_num_) break;
        double depth = feature.depth_estimation;
        if (feature.obs_vec.size() < TRACK_TIME_THRESHOLD || depth == INVALID_DEPTH) {
            continue;
        }
        double inv_depth = para_features_[err_num];
        if (inv_depth > 0) {
            feature.depth_estimation = 1.0 / inv_depth;
        } else {
            feature.depth_estimation = INVALID_DEPTH;
        }

        err_num++;
    }
}

std::set<int> StateEstimator::outliersReprojection() {
    std::set<int> outlier_ids;

    reproj_error_sum_ = 0.0;
    int num = 0;
    for (auto& [id, feature] : feature_manager_->get_all_features()) {
        int err_num_ = 0;
        double err_reprojection = 0;
        const std::vector<Observation>& obs_vec = feature.obs_vec;
        if (feature.depth_estimation == INVALID_DEPTH || obs_vec.size() < TRACK_TIME_THRESHOLD) {
            continue;
        }

        int frame_i = feature.start_frame_id;
        int frame_j = frame_i - 1;
        Eigen::Vector3d pt_i = obs_vec[0].uv;
        // double depth = 1.0 / para_features_[err_num_];
        // feature.depth_estimation = depth;
        double depth = feature.depth_estimation;
        for (const auto& obs : obs_vec) {
            ++frame_j;
            if (frame_i != frame_j) {
                Eigen::Vector3d pt_j = obs.uv;
                err_reprojection += computeErrorReprojection(frame_i, frame_j, pt_i, pt_j, depth);
            }
        }

        reproj_error_sum_ += err_reprojection;
        ++err_num_;
        double averger_err = err_reprojection / err_num_;
        if (averger_err > error_thres_) {
            outlier_ids.insert(id);
        }

        num += err_num_;
    }
    reproj_error_sum_ /= num;
    return outlier_ids;
}
void StateEstimator::slideWindow() {
    for (int i = 1; i <= MAX_WINDOW_SIZE; ++i) {
        windows[i - 1] = windows[i];
    }

    feature_manager_->removeOldestFeatures(windows);
}

void StateEstimator::initImuPose(const std::vector<Eigen::Vector3d>& acc_vec) {
    Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
    for (const auto& acc : acc_vec) avg_acc += acc;
    avg_acc /= static_cast<double>(acc_vec.size());
    std::cout << "Estimated gravity direction: " << avg_acc.transpose() << std::endl;

    Eigen::Vector3d normalized_gravity = avg_acc.normalized();
    Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
    Eigen::Matrix3d R_ =
        Eigen::Quaterniond::FromTwoVectors(normalized_gravity, z_axis).toRotationMatrix();
    Eigen::Vector3d n = R_.col(0);
    double y = atan2(n(1), n(0));
    Eigen::Matrix3d Rz;
    Rz << cos(-y), -sin(-y), 0, sin(-y), cos(-y), 0, 0, 0, 1;
    R_ = Rz * R_;
    std::cout << R_ << std::endl;
    windows[0].R = R_;
    R = R_;
    init_imu_pose_ = true;
}
}  // namespace stereo_vins
