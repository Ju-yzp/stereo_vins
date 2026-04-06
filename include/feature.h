#ifndef STEREO_VINS_FEATURE_H_
#define STEREO_VINS_FEATURE_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <cmath>
#include <cstdlib>
#include <vector>

#include <slideWindow.h>
#include <utility.h>

namespace stereo_vins {

struct Feature {
    Feature(int start_frame_id_)
        : start_frame_id(start_frame_id_), depth_estimation(INVALID_DEPTH) {}

    Feature() : start_frame_id(-1), depth_estimation(INVALID_DEPTH) {}

    void add_newObservation(const Observation& new_observation) {
        obs_vec.emplace_back(new_observation);
    }

    bool removeOldestFrame(
        const Eigen::Matrix3d& R0, const Eigen::Vector3d& P0, const Eigen::Matrix3d& R1,
        const Eigen::Vector3d& P1) {
        if (start_frame_id == 0) {
            if (obs_vec.size() > 1) {
                Eigen::Vector3d project_pt =
                    R1.transpose() * ((R0 * obs_vec[0].uv * depth_estimation + P0) - P1);
                depth_estimation = INVALID_DEPTH;
                if (project_pt.z() > MIN_DISTANCE && project_pt.z() < MAX_DISTANCE) {
                    depth_estimation = project_pt.z();
                }
                obs_vec.erase(obs_vec.begin());
                return false;
            } else {
                return true;
            }
        }
        --start_frame_id;
        return false;
    }

    bool removeNewestFrame(int frame_count) {
        if (start_frame_id == frame_count) {
            return true;
        }
        if (start_frame_id + (int)obs_vec.size() - 1 == frame_count) {
            obs_vec.pop_back();
        }
        return obs_vec.empty();
    }

    // TODO:
    // 使用双目初始化路标深度，在左右目像素误差较大时，仍然能求解出物理上成立的点，但是距离极为不可信
    // 如果使用硬截断双目求深度的最大深度，也不能避免接下来的三角化出的点不符合事实
    void initFirstObservationDepth(
        SlideWindow* windows, const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic,
        const Eigen::Matrix3d& ric1, const Eigen::Vector3d& tic1, bool enable_triangle) {
        if (depth_estimation != INVALID_DEPTH) {
            return;
        }

        if (obs_vec[0].is_stereo) {
            Eigen::Matrix<double, 3, 4> pose0 = Eigen::Matrix<double, 3, 4>::Identity();
            Eigen::Matrix<double, 3, 4> pose1;
            pose1.block<3, 3>(0, 0) = ric1.transpose() * ric;
            pose1.block<3, 1>(0, 3) = ric1.transpose() * (tic - tic1);
            Eigen::Matrix4d A;
            Eigen::Vector3d uv = obs_vec[0].uv;
            Eigen::Vector3d uv_r = obs_vec[0].uv_r;
            A.row(0) = uv.x() * pose0.row(2) - pose0.row(0);
            A.row(1) = uv.y() * pose0.row(2) - pose0.row(1);
            A.row(2) = uv_r.x() * pose1.row(2) - pose1.row(0);
            A.row(3) = uv_r.y() * pose1.row(2) - pose1.row(1);
            Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
            Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);
            Eigen::Vector3d point_3d = homogeneous_point.head<3>() / homogeneous_point.w();
            if (point_3d.z() > MIN_DISTANCE && point_3d.z() < MAX_DISTANCE) {
                Eigen::Vector3d new_pt =
                    pose1.block<3, 3>(0, 0) * point_3d + pose1.block<3, 1>(0, 3);
                if (new_pt.z() > MIN_DISTANCE && new_pt.z() < MAX_DISTANCE) {
                    depth_estimation = point_3d.z();
                    return;
                }
                obs_vec[0].is_stereo = false;
            }
            obs_vec[0].is_stereo = false;
        }
        if (enable_triangle) {
            int track_time = static_cast<int>(obs_vec.size());
            if (track_time > 1) {
                Eigen::MatrixXd A(2 * track_time, 4);
                Eigen::Matrix3d reference_r = windows[start_frame_id].R * ric;
                Eigen::Vector3d reference_t =
                    windows[start_frame_id].R * tic + windows[start_frame_id].P;

                int cur_frame_id = start_frame_id;

                int row = 0;
                for (auto& observation : obs_vec) {
                    Eigen::Matrix3d cur_r = windows[cur_frame_id].R * ric;
                    Eigen::Vector3d cur_t = windows[cur_frame_id].R * tic + windows[cur_frame_id].P;

                    Eigen::Matrix3d dr = cur_r.transpose() * reference_r;
                    Eigen::Vector3d dt = cur_r.transpose() * (reference_t - cur_t);
                    Eigen::Matrix<double, 3, 4> pose;
                    pose.block(0, 0, 3, 3) = dr;
                    pose.block(0, 3, 3, 1) = dt;
                    Eigen::Vector3d direction = observation.uv;

                    A.row(row++) = direction.x() * pose.row(2) - pose.row(0);
                    A.row(row++) = direction.y() * pose.row(2) - pose.row(1);
                    ++cur_frame_id;
                }

                Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
                Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);
                double depth = (homogeneous_point.head<3>() / homogeneous_point.w()).z();
                if (depth > MIN_DISTANCE && depth < MAX_DISTANCE) {
                    depth_estimation = depth;
                    return;
                }
            }
        }
    }
    int start_frame_id;
    std::vector<Observation> obs_vec;
    double depth_estimation;
};
}  // namespace stereo_vins
#endif
