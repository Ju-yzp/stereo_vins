#ifndef STEREO_VINS_PARAMS_H_
#define STEREO_VINS_PARAMS_H_

#include <tools/paramsParser.h>
#include <Eigen/Core>
#include <cstddef>
#include <map>
#include <opencv2/opencv.hpp>
namespace stereo_vins {
struct Params {
    explicit Params(const std::string config_file) {
        ParamsParser param_parser(config_file);
        rows = param_parser.as<int>("rows");
        cols = param_parser.as<int>("cols");
        grid_rows = param_parser.as<int>("grid_rows");
        grid_cols = param_parser.as<int>("grid_cols");
        border_size = param_parser.as<int>("border_size");
        max_features = param_parser.as<int>("max_features");
        dist_threshold = param_parser.as<double>("dist_threshold");
        mask_radius = param_parser.as<double>("mask_radius");
        flow_back = param_parser.as<bool>("flow_back");
        max_move_dist = param_parser.as<double>("max_move_dist");
        enable_gray_gradient = param_parser.as<bool>("enable_gray_gradient");
        enable_statistics = param_parser.as<bool>("enable_statistics");
        new_featurte_ratio = param_parser.as<double>("new_feature_ratio");
        parallax_thres = param_parser.as<double>("parallax_thres");
        reprojection_error_thres = param_parser.as<double>("reprojection_error_thres");
        min_gradient = param_parser.as<double>("min_gradient");
        cam_distort_map[0] = param_parser.as<cv::Mat>("cam0", "distortion_coeffs");
        cam_intrinsic_map[0] = param_parser.as<cv::Mat>("cam0", "intrinsics");
        cam_distort_map[1] = param_parser.as<cv::Mat>("cam1", "distortion_coeffs");
        cam_intrinsic_map[1] = param_parser.as<cv::Mat>("cam1", "intrinsics");
        T_cam_imu_map[0] = param_parser.as<Eigen::Matrix4d>("cam0", "T_cam_imu").inverse();
        T_cam_imu_map[1] = param_parser.as<Eigen::Matrix4d>("cam1", "T_cam_imu").inverse();
        exposureUs = param_parser.as<int>("exposureUs");

        sensitivityIso = param_parser.as<int>("sensitivityIso");
    }

    std::map<size_t, Eigen::Matrix4d> T_cam_imu_map;
    std::map<size_t, cv::Mat> cam_distort_map;
    std::map<size_t, cv::Mat> cam_intrinsic_map;

    // FeatureTracker选项
    int rows, cols;

    int grid_rows, grid_cols;

    int border_size;

    int max_features;

    double dist_threshold;

    double mask_radius;

    bool flow_back;

    double max_move_dist;

    double min_gradient;

    bool enable_gray_gradient;

    bool enable_statistics;

    // FeatureMannager选项
    double new_featurte_ratio;
    double parallax_thres;

    double reprojection_error_thres;

    // OAK-D-LITE配置
    int exposureUs;
    int sensitivityIso;
};
}  // namespace stereo_vins
#endif  // STEREO_VINS_PARAMS_H_
