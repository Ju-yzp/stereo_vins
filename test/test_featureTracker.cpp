#include <Eigen/Core>
#include <chrono>
#include <cstdio>
#include <opencv2/highgui.hpp>

#include <featureTracker.h>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>

using namespace stereo_vins;
int main(int argc, char** argv) {
    using namespace std;
    using namespace std::chrono;
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
    auto sync = pipeline.create<dai::node::Sync>();
    auto xoutStereo = pipeline.create<dai::node::XLinkOut>();

    xoutStereo->setStreamName("stereo");

    monoLeft->setCamera("left");
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setCamera("right");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

    monoLeft->out.link(sync->inputs["left"]);
    monoRight->out.link(sync->inputs["right"]);
    sync->out.link(xoutStereo->input);
    sync->setSyncThreshold(milliseconds(10));

    dai::Device device(pipeline);
    auto stereoQueue = device.getOutputQueue("stereo", 8, false);
    std::string config_file = "/home/adrewn/stereo_vins/config/oak_d_lite.yaml";
    Params params(config_file);
    FeatureTracker feature_tracker(params);

    cv::namedWindow("track_result", cv::WINDOW_AUTOSIZE);
    string save_dir = "/home/adrewn/stereo_vins/calibration_img";

    // while (true) {
    //     auto msgGroup = stereoQueue->get<dai::MessageGroup>();
    //     auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
    //     auto rightFrame = msgGroup->get<dai::ImgFrame>("right");

    //     if (!leftFrame || !rightFrame) continue;

    //     cv::Mat left_img =
    //         cv::Mat(
    //             leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1,
    //             leftFrame->getData().data()) .clone();
    //     cv::Mat right_img = cv::Mat(
    //                             rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1,
    //                             rightFrame->getData().data())
    //                             .clone();

    //     feature_tracker.feedStereo(left_img, right_img);
    //     cv::Mat track_result = feature_tracker.showTrackResult(left_img, right_img);
    //     cv::imshow("track_result", track_result);
    //     cv::waitKey(1);
    // }
    // return 0;
    cv::namedWindow("track_result", cv::WINDOW_AUTOSIZE);

    int save_count = 0;
    cout << "--- 模式说明 ---" << endl;
    cout << "按下 's' 键保存当前左右目图片到 " << save_dir << endl;
    cout << "按下 'q' 键退出程序" << endl;

    while (true) {
        auto msgGroup = stereoQueue->get<dai::MessageGroup>();
        auto leftFrame = msgGroup->get<dai::ImgFrame>("left");
        auto rightFrame = msgGroup->get<dai::ImgFrame>("right");

        if (!leftFrame || !rightFrame) continue;

        cv::Mat left_img(
            leftFrame->getHeight(), leftFrame->getWidth(), CV_8UC1, leftFrame->getData().data());
        cv::Mat right_img(
            rightFrame->getHeight(), rightFrame->getWidth(), CV_8UC1, rightFrame->getData().data());

        // 必须 clone，否则直接操作 SDK 的内存可能导致数据不一致
        cv::Mat left_save = left_img.clone();
        cv::Mat right_save = right_img.clone();

        feature_tracker.feedStereo(left_save, right_save);
        cv::Mat track_result = feature_tracker.showTrackResult(left_save, right_save);

        cv::imshow("track_result", track_result);

        // --- 捕获按键逻辑 ---
        char key = (char)cv::waitKey(1);
        if (key == 's' || key == 'S') {
            string left_path = save_dir + "/left_" + to_string(save_count) + ".png";
            string right_path = save_dir + "/right_" + to_string(save_count) + ".png";

            bool s1 = cv::imwrite(left_path, left_save);
            bool s2 = cv::imwrite(right_path, right_save);

            if (s1 && s2) {
                cout << "[SUCCESS] Saved pair " << save_count << " to " << save_dir << endl;
                save_count++;
            } else {
                cerr << "[ERROR] Failed to save images!" << endl;
            }
        } else if (key == 'q' || key == 'Q') {
            break;
        }
    }
    return 0;
}
