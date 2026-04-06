#include <synchronizer.h>
#include <Eigen/Core>
#include <chrono>
#include <cstdio>
#include <depthai/depthai.hpp>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

int main() {
    dai::Pipeline pipeline;

    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    monoLeft->initialControl.setAutoWhiteBalanceMode(dai::CameraControl::AutoWhiteBalanceMode::OFF);
    monoRight->initialControl.setAutoWhiteBalanceMode(
        dai::CameraControl::AutoWhiteBalanceMode::OFF);

    int exposureUs = 10000;
    int sensitivityIso = 800;
    monoLeft->initialControl.setManualExposure(exposureUs, sensitivityIso);
    monoRight->initialControl.setManualExposure(exposureUs, sensitivityIso);

    auto imu = pipeline.create<dai::node::IMU>();
    auto sync = pipeline.create<dai::node::Sync>();
    auto xoutStereo = pipeline.create<dai::node::XLinkOut>();
    auto xoutImu = pipeline.create<dai::node::XLinkOut>();

    xoutStereo->setStreamName("stereo");
    xoutImu->setStreamName("imu");

    monoLeft->setCamera("left");
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

    monoLeft->out.link(sync->inputs["left"]);
    monoRight->out.link(sync->inputs["right"]);
    sync->out.link(xoutStereo->input);
    sync->setSyncThreshold(std::chrono::milliseconds(10));

    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 400);
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 400);
    imu->setBatchReportThreshold(1);
    imu->setMaxBatchReports(10);
    imu->out.link(xoutImu->input);

    dai::Device device(pipeline);

    stereo_vins::Synchronizer sync_buffer;
    auto stereoQueue = device.getOutputQueue("stereo", 8, false);
    stereoQueue->addCallback([&sync_buffer](std::shared_ptr<dai::ADatatype> data) {
        auto msgGroup = std::dynamic_pointer_cast<dai::MessageGroup>(data);
        if (!msgGroup) return;

        auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
        auto rightFrame = msgGroup->get<dai::ImgFrame>("right");
        if (!leftFrame || !rightFrame) return;

        double ts =
            std::chrono::duration<double>(leftFrame->getTimestamp().time_since_epoch()).count();
        stereo_vins::StereoFrame frame;
        frame.timestamp = ts;
        frame.left_img =
            cv::Mat(
                leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1, leftFrame->getData().data())
                .clone();
        frame.right_img = cv::Mat(
                              rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1,
                              rightFrame->getData().data())
                              .clone();

        sync_buffer.inputStereoFrame(std::move(frame));
    });

    auto imuQueue = device.getOutputQueue("imu", 50, false);
    imuQueue->addCallback([&sync_buffer](std::shared_ptr<dai::ADatatype> data) {
        auto imuData = std::dynamic_pointer_cast<dai::IMUData>(data);
        if (!imuData) return;

        for (auto& packet : imuData->packets) {
            auto& acc = packet.acceleroMeter;
            auto& gyro = packet.gyroscope;

            double ts =
                std::chrono::duration<double>(acc.getTimestamp().time_since_epoch()).count();

            stereo_vins::IMUMeasurement imu_meas;
            imu_meas.timestamp = ts;
            imu_meas.acc = Eigen::Vector3d(acc.x, acc.y, acc.z);
            imu_meas.gyro = Eigen::Vector3d(gyro.x, gyro.y, gyro.z);

            sync_buffer.inputIMUMeasurement(imu_meas);
        }
    });
    std::cout << "Data syncing started. Processing loop running..." << std::endl;

    while (true) {
        stereo_vins::DataBag bag;

        if (sync_buffer.getFinalData(bag, true)) {
            double img_ts = bag.img_buffer->get_timestamp();

            size_t total_imu_samples = 0;
            for (const auto& imu_seg : bag.imu_segments) {
                if (imu_seg && imu_seg->data_buffer_) {
                    total_imu_samples += imu_seg->data_buffer_->size();
                }
            }

            std::cout << "\r[Sync System] Fetched DataBag: Time=" << std::fixed
                      << std::setprecision(3) << img_ts << "s, IMU_Samples=" << std::setw(3)
                      << total_imu_samples << " | Queued: " << std::setw(2) << " " << std::flush;
        }
    }
}
