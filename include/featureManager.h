#ifndef STEREO_VINS_FEATURE_MANAGER_H_
#define STEREO_VINS_FEATURE_MANAGER_H_

// cpp
#include <cmath>
#include <map>
#include <unordered_map>

// eigen
#include <Eigen/Core>
#include <Eigen/Eigen>

// opencv
#include <opencv2/opencv.hpp>

#include <feature.h>
#include <slideWindow.h>
#include <utility.h>

namespace stereo_vins {

class FeatureManager {
public:
    FeatureManager(
        double new_feature_ratio, double parallax_thres, Eigen::Matrix3d ric, Eigen::Vector3d tic,
        Eigen::Matrix3d ric1, Eigen::Vector3d tic1);

    bool checkKeyFrameByParallax(int frame_count, const std::map<int, Observation>& observations);

    void initFramePoseByPNP(int frame_count, SlideWindow* windows);

    void initFeaturesDepth(SlideWindow* windows, bool enable_triangle = true) {
        for (auto& [id, feature] : features_) {
            feature.initFirstObservationDepth(windows, ric_, tic_, ric1_, tic1_, enable_triangle);
        }
    }

    std::unordered_map<int, Feature> get_all_features() { return features_; }

    std::unordered_map<int, Feature> get_all_features() const { return features_; }

    void removeNewestFeatures(int frame_count);

    void removeOldestFeatures(SlideWindow* windows);

    void removeOutliers(std::set<int> outlier_ids);

private:
    inline double computeParallax(const Eigen::Vector2d& pt_i, const Eigen::Vector2d& pt_j) {
        double delta_u = pt_i(0) - pt_j(0);
        double delta_v = pt_i(1) - pt_j(1);
        return std::sqrt((delta_u * delta_u + delta_v * delta_v));
    }

    std::unordered_map<int, Feature> features_;

    double new_feature_ratio_;

    double parallax_thres_;

    Eigen::Matrix3d ric_, ric1_;
    Eigen::Vector3d tic_, tic1_;
};
}  // namespace stereo_vins

#endif  // STEREO_VINS_FEATURE_MANAGER_H
