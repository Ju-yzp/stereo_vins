#ifndef STEREO_VINS_VISULIZER_H_
#define STEREO_VINS_VISULIZER_H_

#include <pangolin/pangolin.h>
#include <Eigen/Dense>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

struct VOUpdateData {
    cv::Mat image;
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> cloud;
    double reproj_error = 0.0;
    bool has_new_data = false;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class VOVisualizer {
public:
    VOVisualizer() : m_quit(false) {
        m_render_thread = std::thread(&VOVisualizer::RenderLoop, this);
    }

    ~VOVisualizer() {
        m_quit = true;
        if (m_render_thread.joinable()) m_render_thread.join();
    }

    void UpdateData(
        const cv::Mat& img, const Eigen::Matrix4d& T_wc, const std::vector<Eigen::Vector3d>& pts,
        double err) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data.image = img.clone();
        m_data.T_wc = T_wc;
        m_data.cloud = pts;
        m_data.reproj_error = err;
        m_data.has_new_data = true;
        m_trajectory.push_back(T_wc);
    }

private:
    void RenderLoop() {
        pangolin::CreateWindowAndBind("VIO Developer Console", 1800, 1000);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const float split_x = 0.45f;

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, -5, 5, 0, 0, 0, 0, 0, 1));

        pangolin::View& d_3d = pangolin::CreateDisplay()
                                   .SetBounds(0.0, 1.0, split_x, 1.0)
                                   .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::View& d_image = pangolin::CreateDisplay().SetBounds(0.5, 1.0, 0.0, split_x);

        pangolin::DataLog error_log;
        error_log.SetLabels({"Reproj Error (px)"});

        pangolin::Plotter plotter(&error_log, 0, 150, 0, 5, 30, 0.5);
        plotter.SetBounds(0.0, 0.5, 0.0, split_x);
        plotter.SetBackgroundColour(pangolin::Colour(0.1f, 0.1f, 0.1f, 1.0f));

        pangolin::DisplayBase().AddDisplay(plotter);

        pangolin::GlTexture image_texture(640, 480, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

        std::deque<float> err_history;

        while (!pangolin::ShouldQuit() && !m_quit) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

            VOUpdateData local_data;
            std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> local_traj;
            bool has_new = false;

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                local_data = m_data;
                local_traj = m_trajectory;
                if (m_data.has_new_data) {
                    has_new = true;
                    m_data.has_new_data = false;
                }
            }

            if (has_new) {
                float current_err = (float)local_data.reproj_error;

                if (std::isnan(current_err) || std::isinf(current_err) || current_err < 0.0f) {
                    current_err = 0.0f;
                }

                error_log.Log(current_err);

                err_history.push_back(current_err);
                if (err_history.size() > 150) {
                    err_history.pop_front();
                }

                float max_err = 5.0f;
                for (float e : err_history) {
                    if (e > max_err) max_err = e;
                }
                max_err *= 1.2f;

                float num = (float)error_log.Samples();
                float x_min = (num > 150.0f) ? num - 150.0f : 0.0f;
                float x_max = (num > 150.0f) ? num : 150.0f;

                plotter.SetView(pangolin::XYRangef(x_min, x_max, 0, max_err));
            }

            d_3d.Activate(s_cam);
            glClear(GL_DEPTH_BUFFER_BIT);

            DrawGrid();
            DrawWorldAxes();

            if (!local_traj.empty()) {
                DrawTrajectory(local_traj);
                DrawCameraFrustum(local_data.T_wc, 0.6f);
                DrawPoints(local_data.cloud);
            }

            if (!local_data.image.empty()) {
                d_image.Activate();
                d_image.SetAspect((float)local_data.image.cols / local_data.image.rows);
                if (image_texture.width != local_data.image.cols ||
                    image_texture.height != local_data.image.rows) {
                    image_texture = pangolin::GlTexture(
                        local_data.image.cols, local_data.image.rows, GL_RGB, false, 0, GL_BGR,
                        GL_UNSIGNED_BYTE);
                }

                image_texture.Upload(local_data.image.data, GL_BGR, GL_UNSIGNED_BYTE);
                glColor3f(1.0, 1.0, 1.0);
                image_texture.RenderToViewportFlipY();
            }

            DrawUIHelpers(split_x);
            pangolin::FinishFrame();
        }
    }

    void DrawGrid() {
        glLineWidth(1.0);
        glColor3f(0.3, 0.3, 0.3);
        glBegin(GL_LINES);
        for (int i = -10; i <= 10; i += 1) {
            glVertex3f(i, -10, 0);
            glVertex3f(i, 10, 0);
            glVertex3f(-10, i, 0);
            glVertex3f(10, i, 0);
        }
        glEnd();
    }

    void DrawWorldAxes() {
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();
    }

    void DrawCameraFrustum(const Eigen::Matrix4d& T_wc, float s) {
        const float w = 0.75f * s;
        const float h = 0.5f * s;
        const float z = s;

        Eigen::Vector4d O_c(0, 0, 0, 1);
        Eigen::Vector4d p1_c(-w, -h, z, 1);
        Eigen::Vector4d p2_c(w, -h, z, 1);
        Eigen::Vector4d p3_c(w, h, z, 1);
        Eigen::Vector4d p4_c(-w, h, z, 1);

        Eigen::Vector3d O_w = (T_wc * O_c).head<3>();
        Eigen::Vector3d p1_w = (T_wc * p1_c).head<3>();
        Eigen::Vector3d p2_w = (T_wc * p2_c).head<3>();
        Eigen::Vector3d p3_w = (T_wc * p3_c).head<3>();
        Eigen::Vector3d p4_w = (T_wc * p4_c).head<3>();

        glLineWidth(2.0f);
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_LINES);

        glVertex3d(O_w.x(), O_w.y(), O_w.z());
        glVertex3d(p1_w.x(), p1_w.y(), p1_w.z());
        glVertex3d(O_w.x(), O_w.y(), O_w.z());
        glVertex3d(p2_w.x(), p2_w.y(), p2_w.z());
        glVertex3d(O_w.x(), O_w.y(), O_w.z());
        glVertex3d(p3_w.x(), p3_w.y(), p3_w.z());
        glVertex3d(O_w.x(), O_w.y(), O_w.z());
        glVertex3d(p4_w.x(), p4_w.y(), p4_w.z());

        glVertex3d(p1_w.x(), p1_w.y(), p1_w.z());
        glVertex3d(p2_w.x(), p2_w.y(), p2_w.z());
        glVertex3d(p2_w.x(), p2_w.y(), p2_w.z());
        glVertex3d(p3_w.x(), p3_w.y(), p3_w.z());
        glVertex3d(p3_w.x(), p3_w.y(), p3_w.z());
        glVertex3d(p4_w.x(), p4_w.y(), p4_w.z());
        glVertex3d(p4_w.x(), p4_w.y(), p4_w.z());
        glVertex3d(p1_w.x(), p1_w.y(), p1_w.z());

        glEnd();
        glLineWidth(1.0f);
    }

    void DrawPoints(const std::vector<Eigen::Vector3d>& cloud) {
        if (cloud.empty()) return;
        glPointSize(2.0);
        glBegin(GL_POINTS);
        glColor3f(0.0, 1.0, 1.0);
        for (const auto& p : cloud) glVertex3d(p.x(), p.y(), p.z());
        glEnd();
    }

    void DrawTrajectory(
        const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& traj) {
        if (traj.size() < 2) return;
        glLineWidth(2.5f);
        glColor3f(0.0, 1.0, 0.0);
        glBegin(GL_LINE_STRIP);
        for (const auto& T : traj) {
            glVertex3d(T(0, 3), T(1, 3), T(2, 3));
        }
        glEnd();
    }

    void DrawUIHelpers(float split_x) {
        pangolin::DisplayBase().Activate();
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glDisable(GL_DEPTH_TEST);
        glLineWidth(2.0f);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_LINES);
        glVertex2f(split_x * 2.0f - 1.0f, -1.0f);
        glVertex2f(split_x * 2.0f - 1.0f, 1.0f);
        glVertex2f(-1.0f, 0.0f);
        glVertex2f(split_x * 2.0f - 1.0f, 0.0f);
        glEnd();
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    std::thread m_render_thread;
    std::mutex m_mutex;
    VOUpdateData m_data;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> m_trajectory;
    bool m_quit;
};

#endif
