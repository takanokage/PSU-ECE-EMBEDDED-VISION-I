#ifndef COMMON_H
#define COMMON_H

#include <string>

#include <opencv2/highgui/highgui.hpp>

namespace common
{
    // suppress displaying images
    extern bool output;

    // Get file folder.
    std::string FileFolder(const std::string& filePath);

    // Get file name with ext.
    std::string FileNameWithExt(const std::string& filePath);

    // Get file name without ext.
    std::string FileName(const std::string& filePath);

    // Check if the file exits.
    bool Exists(const std::string& filePath);

    // Concatenate a directory path with a fileName.
    std::string Concatenate(const std::string& directory, const std::string& fileName);

    // Display the image.
    int Show
        (
        const cv::InputArray& image,
        const std::string& winName,
        const int& cols = 512,
        const int& rows = 512,
        const int& flags = CV_WINDOW_NORMAL,
        const int& posx = 0,
        const int& posy = 0
        );

    // Display the image.
    int Show
        (
        const cv::InputArray& image,
        const std::string& winName,
        const cv::Size& size = cv::Size(512, 512),
        const int& flags = CV_WINDOW_NORMAL,
        const cv::Point& pos = cv::Point(0, 0)
        );

    // Save image to disk.
    bool Save(const cv::InputArray& image, const std::string& dstFile);

    // Read the image from the disk.
    cv::Mat ReadImage(const std::string& imageFile, const std::string& winName, const bool& show = true);
}

#endif