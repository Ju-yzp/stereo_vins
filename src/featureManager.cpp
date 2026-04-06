#include <featureManager.h>
#include <utility.h>

// eigen
#include <Eigen/Core>

// cpp
#include <cassert>
#include <cmath>
#include <vector>

// opencv
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>

namespace stereo_vins {
FeatureManager::FeatureManager(
    double new_feature_ratio, double parallax_thres, Eigen::Matrix3d ric, Eigen::Vector3d tic,
    Eigen::Matrix3d ric1, Eigen::Vector3d tic1)
    : new_feature_ratio_(new_feature_ratio),
      parallax_thres_(parallax_thres),
      ric_(ric),
      tic_(tic),
      ric1_(ric1),
      tic1_(tic1) {}

bool FeatureManager::checkKeyFrameByParallax(
    int frame_count, const std::map<int, Observation>& observations) {
    double new_feature_num = 0;
    double old_feature_num = 0;
    double parallax_sum = 0;

    for (auto& [id, observation] : observations) {
        auto it = features_.find(id);
        if (it != features_.end()) {
            parallax_sum += computeParallax(it->second.obs_vec.back().uv_raw, observation.uv_raw);
            it->second.add_newObservation(observation);
            ++old_feature_num;
        } else {
            features_[id] = Feature(frame_count);
            features_[id].add_newObservation(observation);
            ++new_feature_num;
        }
    }

    double actual_new_ratio = new_feature_num / (new_feature_num + old_feature_num);
    double avg_parallax = (old_feature_num > 0) ? (parallax_sum / old_feature_num) : 0;

    bool cond1 = (old_feature_num == 0);
    bool cond2 = (actual_new_ratio > new_feature_ratio_);
    bool cond3 = (avg_parallax > parallax_thres_);

    return cond1 || cond3;
}

void FeatureManager::initFramePoseByPNP(int frame_count, SlideWindow* windows) {
    if (frame_count > 0) {
        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;

        std::vector<int> all_ids;
        for (const auto& [id, feature] : features_) {
            int offset = frame_count - feature.start_frame_id;

            double depth = feature.depth_estimation;

            if (offset < 0 ||
                offset + 1 == static_cast<int>(feature.obs_vec.size()) && depth != INVALID_DEPTH) {
                // std::cout<<depth<<std::endl;
                Eigen::Vector3d pt_camera = feature.obs_vec[0].uv * depth;
                Eigen::Vector3d pt_imu = ric_ * pt_camera + tic_;
                Eigen::Vector3d object_pt =
                    windows[feature.start_frame_id].R * pt_imu + windows[feature.start_frame_id].P;

                Eigen::Vector2d image_pt = feature.obs_vec[offset].uv.head<2>();

                object_points.emplace_back(cv::Point3f(object_pt(0), object_pt(1), object_pt(2)));
                image_points.emplace_back(cv::Point2f(image_pt(0), image_pt(1)));
                all_ids.emplace_back(id);
            }
        }

        int pt_num = static_cast<int>(object_points.size());

        Eigen::Matrix3d R = windows[frame_count - 1].R * ric_;
        Eigen::Vector3d P = windows[frame_count - 1].R * tic_ + windows[frame_count - 1].P;

        if (pt_num > 5) {
            cv::Mat R_cv, rvec, tvec;
            R.transposeInPlace();
            P = (-R * P).eval();
            cv::eigen2cv(R, R_cv);
            cv::eigen2cv(P, tvec);
            cv::Rodrigues(R_cv, rvec);

            cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
            std::vector<int> inliers;
            bool success = cv::solvePnPRansac(
                object_points, image_points, K, cv::Mat(), rvec, tvec, true, 30, 0.01, 0.8, inliers,
                cv::SOLVEPNP_EPNP);

            int inlier_num = static_cast<int>(inliers.size());
            double inlier_ratio = static_cast<double>(inlier_num) / pt_num;

            if (success) {
                cv::Mat R_result_cv;
                cv::Rodrigues(rvec, R_result_cv);
                cv::cv2eigen(R_result_cv, R);
                cv::cv2eigen(tvec, P);

                R.transposeInPlace();
                P = (-R * P).eval();

                windows[frame_count].R = R * ric_.transpose();
                windows[frame_count].P = P - windows[frame_count].R * tic_;
                std::set<int> outlier_ids;
                std::vector<bool> isInlier(all_ids.size(), false);
                for (int idx : inliers) {
                    if (idx >= 0 && idx < isInlier.size()) {
                        isInlier[idx] = true;
                    }
                }

                for (int i = 0; i < all_ids.size(); ++i) {
                    if (!isInlier[i]) {
                        outlier_ids.insert(all_ids[i]);
                    }
                }
                removeOutliers(outlier_ids);
            } else {
                for (const auto& object_point : object_points) {
                    // Handle the case where PnP fails
                    std::cout << "PnP failed for point: " << object_point.x << ", "
                              << object_point.y << ", " << object_point.z << std::endl;
                }
                for (const auto& image_point : image_points) {
                    // Handle the case where PnP fails
                    std::cout << "PnP failed for image point: " << image_point.x << ", "
                              << image_point.y << std::endl;
                }
                std::cout << "PnP inlier ratio too low: " << inlier_ratio << std::endl;
            }
        } else {
            windows[frame_count].R = windows[frame_count - 1].R;
            windows[frame_count].P = windows[frame_count - 1].P;
            std::cout << "Not enough points for PnP: " << pt_num << std::endl;
        }
    }
}

void FeatureManager::removeNewestFeatures(int frame_count) {
    std::erase_if(
        features_, [&](auto& item) { return item.second.removeNewestFrame(frame_count); });
}

void FeatureManager::removeOldestFeatures(SlideWindow* windows) {
    Eigen::Matrix3d R0 = windows[0].R * ric_, R1 = windows[1].R * ric_;
    Eigen::Vector3d P0 = windows[0].P + windows[0].R * tic_,
                    P1 = windows[1].P + windows[1].R * tic_;
    std::erase_if(
        features_, [&](auto& item) { return item.second.removeOldestFrame(R0, P0, R1, P1); });
}

void FeatureManager::removeOutliers(std::set<int> outlier_ids) {
    std::erase_if(features_, [&](const auto& item) { return outlier_ids.contains(item.first); });
}
}  // namespace stereo_vins
