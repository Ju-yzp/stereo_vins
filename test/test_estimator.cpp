#include <estimator.h>
#include <featureTracker.h>
#include <opencv2/core/hal/interface.h>
#include <tools/AEController.h>
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

    string config_file = "/home/adrewn/stereo_vins/config/oak_d_lite.yaml";
    auto params = stereo_vins::Params(config_file);

    dai::Pipeline pipeline;

    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto xInControlL = pipeline.create<dai::node::XLinkIn>();
    auto xInControlR = pipeline.create<dai::node::XLinkIn>();

    xInControlL->setStreamName("control_left");
    xInControlR->setStreamName("control_right");
    xInControlL->out.link(monoLeft->inputControl);
    xInControlR->out.link(monoRight->inputControl);

    monoLeft->setCamera("left");
    monoRight->setCamera("right");
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

    monoLeft->initialControl.setAutoWhiteBalanceMode(dai::CameraControl::AutoWhiteBalanceMode::OFF);
    monoRight->initialControl.setAutoWhiteBalanceMode(
        dai::CameraControl::AutoWhiteBalanceMode::OFF);

    monoLeft->initialControl.setManualExposure(params.exposureUs, params.sensitivityIso);
    monoRight->initialControl.setManualExposure(params.exposureUs, params.sensitivityIso);

    auto imu = pipeline.create<dai::node::IMU>();
    auto xoutImu = pipeline.create<dai::node::XLinkOut>();
    xoutImu->setStreamName("imu");
    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 400);  // 400Hz
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 400);
    imu->setBatchReportThreshold(1);
    imu->setMaxBatchReports(10);
    imu->out.link(xoutImu->input);

    auto sync = pipeline.create<dai::node::Sync>();
    auto xoutStereo = pipeline.create<dai::node::XLinkOut>();
    xoutStereo->setStreamName("stereo");
    monoLeft->out.link(sync->inputs["left"]);
    monoRight->out.link(sync->inputs["right"]);
    sync->out.link(xoutStereo->input);
    sync->setSyncThreshold(milliseconds(10));

    dai::Device device(pipeline);
    auto stereoQueue = device.getOutputQueue("stereo", 8, false);
    auto imuQueue = device.getOutputQueue("imu", 50, false);
    auto controlQueueL = device.getInputQueue("control_left");
    auto controlQueueR = device.getInputQueue("control_right");

    stereo_vins::StateEstimator estimator(params);
    stereo_vins::FeatureTracker feature_tracker(params);
    VOVisualizer visualizer;

    auto ae_cb_l = [&](double exp_us) {
        dai::CameraControl ctrl;
        ctrl.setManualExposure((int)exp_us, params.sensitivityIso);
        controlQueueL->send(ctrl);
    };
    auto ae_cb_r = [&](double exp_us) {
        dai::CameraControl ctrl;
        ctrl.setManualExposure((int)exp_us, params.sensitivityIso);
        controlQueueR->send(ctrl);
    };

    stereo_vins::AutoExposureController aeL(ae_cb_l), aeR(ae_cb_r);
    aeL.setTargetLuma(75.0);
    aeR.setTargetLuma(75.0);

    imuQueue->addCallback([&](shared_ptr<dai::ADatatype> data) {
        auto imuData = dynamic_pointer_cast<dai::IMUData>(data);
        if (!imuData) return;
        for (auto& packet : imuData->packets) {
            auto& acc = packet.acceleroMeter;
            auto& gyro = packet.gyroscope;
            estimator.inputIMUData(Eigen::Vector3d(acc.x, acc.y, acc.z));
        }
    });

    while (true) {
        auto msgGroup = stereoQueue->get<dai::MessageGroup>();
        auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
        auto rightFrame = msgGroup->get<dai::ImgFrame>("right");
        if (!leftFrame || !rightFrame) continue;

        double ts = leftFrame->getTimestamp().time_since_epoch().count() / 1e9;
        double left_es = static_cast<double>(leftFrame->getExposureTime().count());
        double right_es = static_cast<double>(rightFrame->getExposureTime().count());
        cv::Mat l_img =
            cv::Mat(
                leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1, leftFrame->getData().data())
                .clone();
        cv::Mat r_img = cv::Mat(
                            rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1,
                            rightFrame->getData().data())
                            .clone();

        aeL.changeExposureTime(l_img, ts, left_es, "left");
        aeR.changeExposureTime(r_img, ts, right_es, "right");

        stereo_vins::FeatureFrame feature_frame = feature_tracker.feedStereo(l_img, r_img);
        cv::Mat track_result = feature_tracker.showTrackResult(l_img, r_img);

        estimator.processMeasurement(feature_frame);

        visualizer.UpdateData(
            track_result, estimator.get_current_pose(), estimator.get_pointcloud(),
            estimator.getReprojectionError());
    }

    return 0;
}
