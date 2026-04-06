#ifndef STEREO_VINS_FEATURE_TRACKER_H_
#define STEREO_VINS_FEATURE_TRACKER_H_

// opencv
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Eigen>

#include <feature.h>
#include <tools/params.h>

// cpp
#include <utility.h>
#include <cstddef>
#include <optional>
#include <vector>

namespace stereo_vins {
class FeatureTracker {
public:
    FeatureTracker(const Params params);

    FeatureFrame feedStereo(const cv::Mat& left_img, const cv::Mat& right_img);

    void reset();

    cv::Mat showTrackResult(const cv::Mat& left_img, const cv::Mat& right_img);

private:
    // 我们把右目也视为左目处于不同时刻的状态，复用单目光流匹配的函数
    void monocularMatching(
        const cv::Mat& prev_img, const cv::Mat& cur_img, std::vector<cv::Point2f>& prev_pts,
        std::vector<cv::Point2f>& cur_pts, std::vector<size_t>& prev_ids,
        std::vector<size_t>& cur_ids);

    // 在旧特征点消失后补充新特征点
    void replenishKeypoints(
        const cv::Mat& mask, std::vector<cv::Point2f>& new_pts, std::vector<cv::Point2f>& grad);

    // 生成提取新特征点所用的掩码
    void setMask();
    inline double computeMoveDist(const cv::Point2f& p1, const cv::Point2f& p2) {
        return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }
    inline bool isInsideImage(cv::Point2f pt) {
        return pt.x >= 0 && pt.x < cols_ && pt.y >= 0 && pt.y < rows_;
    }

    std::optional<cv::Point2f> undistort(size_t id, const cv::Point2f& uv_raw);

    bool isBadStereoPoint(Eigen::Vector3d uv, Eigen::Vector3d uv_r);
    // 图像大小
    int rows_, cols_;

    // 棋盘格的大小
    int grid_rows_, grid_cols_;

    // 禁止提取特征点的边界
    int border_size_;

    int max_features_;

    // 两个特征点之间的最小距离，用于减少密集程度
    double dist_threshold_;

    // 已存在特征点的附近添加掩码，避免密集提取
    double mask_radius_;

    double min_gradient_;

    // 特征点id,单调递增
    size_t feature_id_;

    // 反向追踪
    bool flow_back_;

    // 开启反向追踪后，跟正向追踪所允许的最大移动距离
    double max_move_dist_;

    // 特征点集合
    std::vector<cv::Point2f> prev_pts_, cur_pts_, cur_rpts_;

    // 特征点对应的id,一对一
    std::vector<size_t> prev_ids_, cur_ids_, right_ids_;

    cv::Mat cur_img_, prev_img_;

    // 提取新特征点的掩码
    cv::Mat exist_mask_;

    std::vector<bool> occupy_mask_;

    // 相机内参
    std::map<size_t, cv::Mat> distort_map_;
    std::map<size_t, cv::Mat> k_map_;

    // debug 选项
    bool enable_gray_gradient_;

    bool enable_statistics_;

    std::map<size_t, cv::Point2f> gray_gradient_map_;

    std::map<size_t, size_t> tracked_times_;

    Eigen::Matrix3d ric_, ric1_;
    Eigen::Vector3d tic_, tic1_;

    int grid_x_, grid_y_;
};
}  // namespace stereo_vins
#endif  // STEREO_VINS_FEATURE_TRACKER_H_
