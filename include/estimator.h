#ifndef STEREO_VINS_ESTIMATOR_H_
#define STEREO_VINS_ESTIMATOR_H_

#include <featureManager.h>
#include <slideWindow.h>

// cpp
#include <utility.h>
#include <atomic>
#include <memory>
#include <vector>

// eigen
#include <tools/params.h>
#include <Eigen/Core>
#include <Eigen/Eigen>

// ceres
#include <ceres/ceres.h>

namespace stereo_vins {
class StateEstimator {
public:
    StateEstimator(Params& param) {
        Eigen::Matrix4d Tic = param.T_cam_imu_map[0];
        Eigen::Matrix4d Tic1 = param.T_cam_imu_map[1];
        ric_ = Tic.block<3, 3>(0, 0);
        tic_ = Tic.block<3, 1>(0, 3);
        init_imu_pose_ = false;
        feature_manager_ = std::make_unique<FeatureManager>(
            param.new_featurte_ratio, param.parallax_thres, ric_, tic_, Tic1.block<3, 3>(0, 0),
            Tic1.block<3, 1>(0, 3));
        error_thres_ = param.reprojection_error_thres;
        frame_count_ = 0;
    }

    void processMeasurement(FeatureFrame& feature_frame);

    void optimize();

    void stateToParams();

    void paramsToState();

    std::set<int> outliersReprojection();

    void slideWindow();

    // void fast_predict_state(const Measurement& measurement);

    double getReprojectionError() const { return reproj_error_sum_; }

    Eigen::Matrix4d get_current_pose() {
        Eigen::Matrix4d pose;
        pose.block<3, 3>(0, 0) = windows[frame_count_ - 1].R * ric_;
        pose.block<3, 1>(0, 3) = windows[frame_count_ - 1].P + windows[frame_count_ - 1].R * tic_;
        return pose;
    }

    void inputIMUData(Eigen::Vector3d acc) {
        static std::vector<Eigen::Vector3d> acc_vec;
        if (!init_imu_pose_.load()) {
            if (acc_vec.size() < 50) {
                acc_vec.emplace_back(acc);
            } else {
                initImuPose(acc_vec);
            }
        }
    }

    std::vector<Eigen::Vector3d> get_pointcloud() {
        std::vector<Eigen::Vector3d> pointcloud;
        for (auto [id, feature] : feature_manager_->get_all_features()) {
            double depth = feature.depth_estimation;
            if (depth != INVALID_DEPTH) {
                int id = feature.start_frame_id;
                Eigen::Vector3d point_i = feature.obs_vec[0].uv * depth;
                Eigen::Vector3d point_in_world =
                    windows[id].R * (ric_ * point_i + tic_) + windows[id].P;
                pointcloud.emplace_back(point_in_world);
            }
        }
        return pointcloud;
    }

private:
    void initImuPose(const std::vector<Eigen::Vector3d>& acc_vec);

    inline double computeErrorReprojection(
        int i, int j, Eigen::Vector3d& pt_i, Eigen::Vector3d& pt_j, double depth) {
        Eigen::Vector3d pt_j_hat =
            ric_.transpose() *
            (windows[j].R.transpose() *
                 (windows[i].R * (ric_ * (pt_i * depth) + tic_) + windows[i].P - windows[j].P) -
             tic_);
        return (pt_j_hat.normalized() - pt_j.normalized()).norm();
    }
    std::unique_ptr<FeatureManager> feature_manager_;

    SlideWindow windows[MAX_WINDOW_SIZE + 1];

    double para_pose_[MAX_WINDOW_SIZE + 1][6];
    double para_features_[MAX_FEATURE_NUM];
    // Eigen::Vector3d Bgs[MAX_WINDOW_SIZE + 1];
    // Eigen::Vector3d Bas[MAX_WINDOW_SIZE + 1];
    // Eigen::Vector3d Vs[MAX_WINDOW_SIZE + 1];

    // imu状态，测试用的，给vo提供尺度
    Eigen::Vector3d P;
    Eigen::Matrix3d R;
    Eigen::Vector3d V;
    Eigen::Vector3d Bg;
    Eigen::Vector3d Ba;

    int frame_count_;

    double error_thres_;

    std::atomic<bool> init_imu_pose_;

    Eigen::Matrix3d ric_;
    Eigen::Vector3d tic_;

    int feature_num_;

    double reproj_error_sum_;
};
}  // namespace stereo_vins
#endif
