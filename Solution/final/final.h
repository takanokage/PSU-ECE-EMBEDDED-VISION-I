#ifndef FINAL_H
#define FINAL_H

#include "common.h"
#include "idx1.h"
#include "idx3.h"
#include "KNN.h"

#include <opencv2/features2d.hpp>

// Scale factors used to display digits
const double SCALE_FACTOR = 10;

// window coordinates
const int POS_X = 10;
const int POS_Y = 10;

enum class KEY
{
    ESC = 27,
    ELSE
};

// Init rand engine.
void Seed();

// Command line help.
void CmdHelp();

// Get a random integer value between 0 and vmax.
int32_t rand(const int32_t& vmax, const int32_t& vmin = 0);

// Display a digit in its own window.
KEY ShowDigit(const std::string& winName, const idx3& mnistTrnImages, const int& index);
KEY ShowDigit(const std::string& winName, const cv::Mat& images, const int& rows, const int& cols, const int& index);

// Display randomly picked digits, one by one, until the Esc key is pressed.
void DisplayDigits
(
    const idx3& mnistImages,
    const idx1& mnistLabels,
    const int& rows,
    const int& cols
);

// Get MNIST images as single cv::Mat. Each row is one MNIST image.
cv::Mat GetData(const idx3& images);

// Get MNIST labels as single cv::Mat. Each row is one MNIST label.
cv::Mat GetData(const idx1& labels);

// Read the test data from the disk.
void InitTestData
(
    const idx3& mnistTstImages,
    const idx1& mnistTstLabels,
    cv::Mat& tstImages,
    cv::Mat& tstLabels,
    const int& rndTestSize = 0
);

// Perform the single digit detection test.
void SingleDigitTest
(
    const idx3& mnistTrnImages,
    const idx1& mnistTrnLabels,
    const idx3& mnistTstImages,
    const idx1& mnistTstLabels,
    const int& K,
    const int& testSize,
    const bool displayPredictions = false
);

// Perform the digit detection in a synthetic image test.
void SyntheticTest
(
    const idx3& mnistTrnImages,
    const idx1& mnistTrnLabels,
    const idx3& mnistTstImages,
    const idx1& mnistTstLabels,
    const int& K,
    const int& testSize,
    const int& batchSize,
    const bool& displayPredictions = false,
    const bool& suppress = false
);

// Remove small boxes.
void RejectSmallBoxes
(
    std::vector<cv::Rect>& boxes,
    const int& rows,
    const int& cols
);

// Fuse overlapping boxes.
void FuseOverlapping
(
    std::vector<cv::Rect>& boxes,
    const int& rows, // digit rows
    const int& cols  // digit cols
);

// Resize the boxes to match MNIST digit dimensions.
// Center the boxes around the original contents.
void ReSizeCenter
(
    const cv::Rect& bounds,
    std::vector<cv::Rect>& boxes,
    const int& rows, // digit rows
    const int& cols  // digit cols
);

// Generate an image with randomly placed digits. No overlapping.
cv::Mat GetSyntheticImage
(
    const int& nrDigits,
    const idx1& mnistTstLabels,
    const idx3& mnistTstImages,
    std::vector<cv::Point>& rndLocations,
    std::vector<float>& rndLabels
);

// Find bounding boxes around potential digits.
std::vector<cv::Rect> FindBoundingBoxes(const cv::Mat& synthetic);

// Get the expected labels for the digits in the synthetic image.
cv::Mat GetTestLabels
(
    const std::vector<cv::Rect>& boxes,
    const std::vector<cv::Point>& rndLocations,
    const std::vector<float>& rndLabels,
    const int& rows,
    const int& cols
);

// Display the potential digits identified in the synthetic image.
void DisplayFoundDigits(const cv::Mat& synthetic, const std::vector<cv::Rect>& boxes, const std::string& winName);

// Get the test images in the format expected by the cv::KNearest class.
cv::Mat GetTestImages
(
    const cv::Mat& synthetic,
    const std::vector<cv::Rect>& boxes,
    const int& rows,
    const int& cols
);

// Display test statistics.
void DisplayStatistics
(
    const double& accuracy,
    const int& testSize,
    const int& K,
    const double& duration,
    const bool& header = true
);

// Display predicted values side by side with expected/true values.
void DisplayPredictions(const KNN& knn, const int& testSize);

#endif