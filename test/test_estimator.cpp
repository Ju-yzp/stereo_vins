#include <estimator.h>
#include <featureTracker.h>
#include <opencv2/core/hal/interface.h>
#include <tools/params.h>
#include <tools/paramsParser.h>
#include <tools/visualizer.h>
#include <Eigen/Core>
#include <chrono>
#include <cstdio>
#include <depthai/depthai.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char** argv) {
    using namespace std;
    using namespace std::chrono;

    std::string config_file = "/home/adrewn/stereo_vins/config/oak_d_lite.yaml";
    auto params = stereo_vins::Params(config_file);

    dai::Pipeline pipeline;

    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    monoLeft->initialControl.setAutoWhiteBalanceMode(dai::CameraControl::AutoWhiteBalanceMode::OFF);
    monoRight->initialControl.setAutoWhiteBalanceMode(
        dai::CameraControl::AutoWhiteBalanceMode::OFF);

    int exposureUs = params.exposureUs;
    int sensitivityIso = params.sensitivityIso;
    monoLeft->initialControl.setManualExposure(exposureUs, sensitivityIso);
    monoRight->initialControl.setManualExposure(exposureUs, sensitivityIso);

    auto imu = pipeline.create<dai::node::IMU>();
    auto sync = pipeline.create<dai::node::Sync>();
    auto xoutStereo = pipeline.create<dai::node::XLinkOut>();
    auto xoutImu = pipeline.create<dai::node::XLinkOut>();

    xoutStereo->setStreamName("stereo");
    xoutImu->setStreamName("imu");

    monoLeft->setCamera("left");
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    monoRight->setCamera("right");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

    monoLeft->out.link(sync->inputs["left"]);
    monoRight->out.link(sync->inputs["right"]);
    sync->out.link(xoutStereo->input);
    sync->setSyncThreshold(milliseconds(10));

    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 400);
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 400);
    imu->setBatchReportThreshold(1);
    imu->setMaxBatchReports(10);
    imu->out.link(xoutImu->input);

    dai::Device device(pipeline);

    auto stereoQueue = device.getOutputQueue("stereo", 8, false);
    auto imuQueue = device.getOutputQueue("imu", 50, false);
    // stereoQueue->addCallback([&sync_buffer](shared_ptr<dai::ADatatype> data) {
    //     auto msgGroup = dynamic_pointer_cast<dai::MessageGroup>(data);
    //     if (!msgGroup) return;

    //     auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
    //     auto rightFrame = msgGroup->get<dai::ImgFrame>("right");
    //     if (!leftFrame || !rightFrame) return;

    //     double ts = duration<double>(leftFrame->getTimestamp().time_since_epoch()).count();

    //     cv::Mat l_img =
    //         cv::Mat(
    //             leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1,
    //             leftFrame->getData().data()) .clone();
    //     cv::Mat r_img = cv::Mat(
    //                         rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1,
    //                         rightFrame->getData().data())
    //                         .clone();

    //     sync_buffer.inputImageData(l_img, r_img, ts);
    // });
    stereo_vins::StateEstimator estimator(params);
    imuQueue->addCallback([&](shared_ptr<dai::ADatatype> data) {
        auto imuData = dynamic_pointer_cast<dai::IMUData>(data);
        if (!imuData) return;

        for (auto& packet : imuData->packets) {
            auto& acc = packet.acceleroMeter;
            auto& gryo = packet.gyroscope;
            double ts = duration<double>(acc.getTimestamp().time_since_epoch()).count();

            // stereo_vins::ImuMeasurement imu_meas;
            // imu_meas.ts = ts;
            // imu_meas.acc = Eigen::Vector3d(acc.x, acc.y, acc.z);
            // imu_meas.gryo = Eigen::Vector3d(gryo.x, gryo.y, gryo.z);
            // sync_buffer.inputImuData(imu_meas);
            estimator.inputIMUData(Eigen::Vector3d(acc.x, acc.y, acc.z));
        }
    });

    stereo_vins::FeatureTracker feature_tracker(params);
    VOVisualizer visualizer;
    while (true) {
        auto msgGroup = stereoQueue->get<dai::MessageGroup>();
        auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
        auto rightFrame = msgGroup->get<dai::ImgFrame>("right");

        if (!leftFrame || !rightFrame) continue;

        cv::Mat left_img =
            cv::Mat(
                leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1, leftFrame->getData().data())
                .clone();
        cv::Mat right_img = cv::Mat(
                                rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1,
                                rightFrame->getData().data())
                                .clone();

        stereo_vins::FeatureFrame feature_frame = feature_tracker.feedStereo(left_img, right_img);
        cv::Mat track_result = feature_tracker.showTrackResult(left_img, right_img);
        estimator.processMeasurement(feature_frame);
        Eigen::Matrix4d pose = estimator.get_current_pose();
        visualizer.UpdateData(
            track_result, pose, estimator.get_pointcloud(), estimator.getReprojectionError());
    }
}

// #include <estimator.h>
// #include <featureTracker.h>
// #include <opencv2/core/hal/interface.h>
// #include <tools/params.h>
// #include <tools/paramsParser.h>
// #include <tools/visualizer.h>
// #include <Eigen/Core>
// #include <chrono>
// #include <cstdio>
// #include <depthai/depthai.hpp>
// #include <memory>
// #include <opencv2/opencv.hpp>
// #include <vector>

// // --- AE 独立调节辅助类 ---
// struct BrightnessHistogram {
//     std::vector<int> histogram;
//     int mean_luma = 0;
//     BrightnessHistogram() { histogram.resize(256, 0); }
//     void update(const cv::Mat& gray_img) {
//         if (gray_img.empty()) return;
//         histogram.assign(256, 0);
//         long long sum = 0;
//         for (int y = 0; y < gray_img.rows; ++y) {
//             const uchar* ptr = gray_img.ptr<uchar>(y);
//             for (int x = 0; x < gray_img.cols; ++x) {
//                 int val = ptr[x];
//                 histogram[val]++;
//                 sum += val;
//             }
//         }
//         mean_luma = sum / (gray_img.rows * gray_img.cols);
//     }
// };

// class BSplineExposureController {
// private:
//     float t = 1.0f;
//     int start_ev, target_ev;
//     float step = 0.12f; 
// public:
//     int current_ev;
//     BSplineExposureController(int init_ev) : current_ev(init_ev), start_ev(init_ev), target_ev(init_ev) {}
//     int process(int current_luma, int target_luma) {
//         if (std::abs(current_luma - target_luma) > 5 && t >= 1.0f) {
//             start_ev = current_ev;
//             float ratio = (float)target_luma / std::max(current_luma, 1);
//             target_ev = std::clamp((int)(current_ev * ratio), 100, 30000);
//             t = 0.0f; 
//         }
//         if (t < 1.0f) {
//             t += step;
//             if (t > 1.0f) t = 1.0f;
//             float smooth_t = t * t * (3.0f - 2.0f * t);
//             current_ev = start_ev + (target_ev - start_ev) * smooth_t;
//         }
//         return current_ev;
//     }
// };

// int main(int argc, char** argv) {
//     using namespace std;
//     using namespace std::chrono;

//     std::string config_file = "/home/adrewn/stereo_vins/config/oak_d_lite.yaml";
//     auto params = stereo_vins::Params(config_file);

//     dai::Pipeline pipeline;

//     auto monoLeft = pipeline.create<dai::node::MonoCamera>();
//     auto monoRight = pipeline.create<dai::node::MonoCamera>();
    
//     // --- 修改点：创建两个独立的控制通道 ---
//     auto xInControlL = pipeline.create<dai::node::XLinkIn>();
//     auto xInControlR = pipeline.create<dai::node::XLinkIn>();
//     xInControlL->setStreamName("control_left");
//     xInControlR->setStreamName("control_right");

//     // 物理连接：左通道连左相机，右通道连右相机
//     xInControlL->out.link(monoLeft->inputControl);
//     xInControlR->out.link(monoRight->inputControl);

//     monoLeft->setCamera("left");
//     monoRight->setCamera("right");
//     monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
//     monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

//     int exposureUs = params.exposureUs;
//     int sensitivityIso = params.sensitivityIso;
//     monoLeft->initialControl.setManualExposure(exposureUs, sensitivityIso);
//     monoRight->initialControl.setManualExposure(exposureUs, sensitivityIso);

//     auto imu = pipeline.create<dai::node::IMU>();
//     auto sync = pipeline.create<dai::node::Sync>();
//     auto xoutStereo = pipeline.create<dai::node::XLinkOut>();
//     auto xoutImu = pipeline.create<dai::node::XLinkOut>();

//     xoutStereo->setStreamName("stereo");
//     xoutImu->setStreamName("imu");

//     monoLeft->out.link(sync->inputs["left"]);
//     monoRight->out.link(sync->inputs["right"]);
//     sync->out.link(xoutStereo->input);
//     sync->setSyncThreshold(milliseconds(10));

//     imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 400);
//     imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 400);
//     imu->out.link(xoutImu->input);

//     dai::Device device(pipeline);
//     auto stereoQueue = device.getOutputQueue("stereo", 8, false);
//     auto imuQueue = device.getOutputQueue("imu", 50, false);
    
//     // --- 修改点：获取两个独立的控制队列 ---
//     auto controlQueueL = device.getInputQueue("control_left");
//     auto controlQueueR = device.getInputQueue("control_right");

//     stereo_vins::StateEstimator estimator(params);
//     imuQueue->addCallback([&](shared_ptr<dai::ADatatype> data) {
//         auto imuData = dynamic_pointer_cast<dai::IMUData>(data);
//         if (!imuData) return;
//         for (auto& packet : imuData->packets) {
//             auto& acc = packet.acceleroMeter;
//             estimator.inputIMUData(Eigen::Vector3d(acc.x, acc.y, acc.z));
//         }
//     });

//     stereo_vins::FeatureTracker feature_tracker(params);
//     VOVisualizer visualizer;

//     // --- 初始化 AE 参数 ---
//     BrightnessHistogram histL, histR;
//     BSplineExposureController aeL(exposureUs), aeR(exposureUs);
//     int target_luma = 105;
//     int lastL = exposureUs, lastR = exposureUs;

//     while (true) {
//         auto msgGroup = stereoQueue->get<dai::MessageGroup>();
//         auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
//         auto rightFrame = msgGroup->get<dai::ImgFrame>("right");
//         if (!leftFrame || !rightFrame) continue;

//         cv::Mat l_img = cv::Mat(leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1, leftFrame->getData().data()).clone();
//         cv::Mat r_img = cv::Mat(rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1, rightFrame->getData().data()).clone();

//         // --- 1. 左目调节 ---
//         histL.update(l_img);
//         int nextL = aeL.process(histL.mean_luma, target_luma);
//         if (std::abs(nextL - lastL) > 2) {
//             dai::CameraControl ctrl;
//             ctrl.setManualExposure(nextL, sensitivityIso);
//             controlQueueL->send(ctrl); // 发送到左队列
//             lastL = nextL;
//         }

//         // --- 2. 右目调节 ---
//         histR.update(r_img);
//         int nextR = aeR.process(histR.mean_luma, target_luma);
//         if (std::abs(nextR - lastR) > 2) {
//             dai::CameraControl ctrl;
//             ctrl.setManualExposure(nextR, sensitivityIso);
//             controlQueueR->send(ctrl); // 发送到右队列
//             lastR = nextR;
//         }

//         // --- VINS 处理 ---
//         stereo_vins::FeatureFrame feature_frame = feature_tracker.feedStereo(l_img, r_img);
//         cv::Mat track_result = feature_tracker.showTrackResult(l_img, r_img);
        
//         char info[128];
//         snprintf(info, sizeof(info), "L_EV: %d | R_EV: %d", nextL, nextR);
//         cv::putText(track_result, info, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);

//         estimator.processMeasurement(feature_frame);
//         visualizer.UpdateData(track_result, estimator.get_current_pose(), estimator.get_pointcloud(), estimator.getReprojectionError());
//     }
// }
