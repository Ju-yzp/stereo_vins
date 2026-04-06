#include <tools/paramsParser.h>
#include <iostream>

int main() {
    std::string config_file("/home/adrewn/stereo_vins/config/oak_d_lite.yaml");
    stereo_vins::ParamsParser params_parser(config_file);
    cv::Mat intrinsics = params_parser.as<cv::Mat>("cam0", "intrinsics");
    cv::Mat distortion_coeffs = params_parser.as<cv::Mat>("cam0", "distortion_coeffs");
    std::cout << "Intrinsics Matrix:\n"
              << cv::format(intrinsics, cv::Formatter::FMT_PYTHON) << std::endl;
    std::cout << "Distortion_coeffs Vector:\n"
              << cv::format(distortion_coeffs, cv::Formatter::FMT_PYTHON) << std::endl;
}
