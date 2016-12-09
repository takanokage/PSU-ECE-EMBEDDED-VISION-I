#ifndef LAB_03_H
#define LAB_03_H

#include "common.h"

namespace lab03
{
    // Task 01: Histogram based face detection.
    void Task01(const std::string& groupFile, const std::string& templateFile);

    // Task 01: attempt to improve the algorithm.
    void Task01b(const std::string& groupFile, const std::string& templateFile);

    // Task 02: Template matching.
    void Task02(const std::string& groupFile, const std::string& templateFile);

    // Task 03: Find contours.
    void Task03(const std::string& imageFile);

    // Extract an image fragment and save it to disk.
    void ExtractROI(int event, int x, int y, int flags, void* userdata);

    // Face capture on mouse left button click. 
    void CaptureFaces(const std::string& imageFile);

    // Color reduce an image.
    static cv::Mat ColorReduce(const cv::Mat& image, const int& div = 64);
}

#endif