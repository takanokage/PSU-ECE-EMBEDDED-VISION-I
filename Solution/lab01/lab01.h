#ifndef LAB_01_H
#define LAB_01_H

#include "common.h"

namespace lab01
{
    // function pointer to cv::pyrDown and cv::pyrUp
    typedef void(CV_EXPORTS_W pyramid)(class cv::_InputArray const &, class cv::_OutputArray const &, class cv::Size_<int> const &, int);

    // Lab 01 image pyramid
    cv::Mat ImagePyramid(pyramid* func, const cv::Mat& image, const int& scaleSteps, const float& scaleFactor, const std::string& winName, const int& cols, const int& rows);

    // Task 02: image scale down/up.
    void Task02(const std::string& srcFile);

    // Task 03: draw image.
    void Task03();
}

#endif