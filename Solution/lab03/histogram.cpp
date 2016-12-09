#include "histogram.h"

#include <opencv2/opencv.hpp>

using namespace cv;

// ------------------------------------------------------------------------------------------------
// Initialization constructor.
// ------------------------------------------------------------------------------------------------
Histogram::Histogram(const cv::Mat* images, const int & nimages, const int & binX, const int & binY, const int & binZ, const float & rangeMin, const float & rangeMax, const Type & type)
{
    // init input
    this->images = images;
    this->nimages = nimages;

    // init channels used
    this->channels[0] = -1;
    this->channels[1] = -1;
    this->channels[2] = -1;

    // init number of bins
    this->histSize[0] = binX;
    this->histSize[1] = binY;
    this->histSize[2] = binZ;

    // init BRG range
    this->range[0] = rangeMin;
    this->range[1] = rangeMax;

    // init BRG range per channel
    this->ranges[0] = this->range;
    this->ranges[1] = this->range;
    this->ranges[2] = this->range;

    // init histogram type
    this->type = type;

    ComputeHistogram();
}

// ------------------------------------------------------------------------------------------------
// Compute the histogram.
// ------------------------------------------------------------------------------------------------
void Histogram::ComputeHistogram()
{
    switch (type)
    {
    case Type::Default1D:
    {
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;

        dims = 1;

        break;
    }
    case Type::Default3D:
    {
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;

        dims = 3;

        break;
    }
    case Type::Sparse3D:
    case Type::AB2D:
    case Type::Hue1D:
    default: break;
    }

    // OpenCV histogram calculation
    calcHist
    (
        images,
        nimages,
        channels,
        Mat(),
        hist,
        dims,
        histSize,
        (const float**)ranges
    );

    // normalize to 1.0, NORM_L2 is default
    cv::normalize(hist, hist, 1.0);
}

// ------------------------------------------------------------------------------------------------
// Generate the back projection
// ------------------------------------------------------------------------------------------------
Mat Histogram::BackProjection(const Mat& image)
{
    Mat result;

    calcBackProject
    (
        &image,     // input image
        1,          // one image
        channels,   // vector specifying what histogram dimensions belong to what image channels
        hist,       // the histogram we are using
        result,     // the resulting back projection image
        (const float**)ranges,     // the range of values, for each dimension
        255.0       // the scaling factor is chosen such that a histogram value of 1 maps to 255
    );

    return result;
}
