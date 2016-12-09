#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/highgui/highgui.hpp>

class Histogram
{
public:
    // supported histogram types
    enum class Type
    {
        Default1D,
        Default3D,
        Sparse3D,
        AB2D,
        Hue1D
    };

public:
    Histogram
    (
        const cv::Mat* images,
        const int& nimages = 1,
        const int& binX = 256,
        const int& binY = 256,
        const int& binZ = 256,
        const float& rangeMin = 0.0f,
        const float& rangeMax = 255.0f,
        const Type& type = Type::Default3D
    );
    ~Histogram() { };

private:
    // remove default/copy constructor and assignment operator
    Histogram() = delete;
    Histogram(const Histogram& gen) = delete;
    Histogram(const Histogram *const gen) = delete;
    Histogram(Histogram *const gen) = delete;
    Histogram& operator=(const Histogram& gen) = delete;
    Histogram& operator=(const Histogram *const gen) = delete;
    Histogram& operator=(Histogram *const gen) = delete;
private:
    // The input of the histogram calculation.
    const cv::Mat* images;

    // The number of images in the input.
    int nimages = 0;

    // channels used to compute the histogram
    int channels[3];

    // the actual histogram
    cv::MatND hist;

    // The number of dimensions for this histogram.
    int dims = 0;

    // set default values for a color histogram
    int histSize[3];

    // BRG range
    float range[2];

    // BRG range per channel
    float* ranges[3];

    // histogram type.
    Type type;

public:
    // Compute the histogram.
    void ComputeHistogram();

    // Generate the back projection
    cv::Mat BackProjection(const cv::Mat& image);
};

#endif