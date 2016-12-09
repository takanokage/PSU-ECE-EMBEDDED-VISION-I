#ifndef LAB_02_H
#define LAB_02_H

#include "common.h"

namespace lab02
{
    // An enumeration of the morphological operations used in Task02.
    typedef struct _Morphology
    {
        enum class Type
        {
            DILATE,
            ERODE,
            NOP // no operation
        };

        static std::string ToString(const Type& type)
        {
            std::string output = "";

            switch (type)
            {
            case Type::DILATE: return output = "DILATE"; break;
            case Type::ERODE: return output = "ERODE"; break;
            default: return output = "NOP"; break;
            }
        }
    } Morphology;

    // An enumeration of the morphological gradient types used in Task02.
    typedef struct _Gradient
    {
        enum class Type
        {
            DILATED_ERODED,
            ORIGINAL_ERODED,
            DILATED_ORIGINAL,
            NOP // no operation
        };

        static std::string ToString(const Type& type)
        {
            std::string output = "";

            switch (type)
            {
            case Type::DILATED_ERODED: return output = "DILATED-ERODED"; break;
            case Type::ORIGINAL_ERODED: return output = "ORIGINAL-ERODED"; break;
            case Type::DILATED_ORIGINAL: return output = "DILATED-ORIGINAL"; break;
            default: return output = "NOP"; break;
            }
        }
    } Gradient;

    // Task 02: Erode/Dilate/Open/Close.
    void Task02(const std::string& t02ImageFile);

    // Task 03: Coffee cup.
    void Task03(const std::string& t03Image1File, const std::string& t03Image2File);

    // Convert to gray.
    cv::Mat ToGray(const cv::Mat& image, const std::string& winName, const bool& show = true);

    // Get the negative.
    cv::Mat ToNeg(const cv::Mat& image, const std::string& winName, const bool& show = true);

    // Convert to a binary image.
    cv::Mat ToBW(const cv::Mat& image, const int& thresholdType, const std::string& winName, const bool& show = true);

    // Perform the Erode/Dilate test for the given Erode/Dilate kernel size and type.
    void ErodeDilateTest(const Morphology::Type& bwOperation, const cv::Mat& bwImage, const Morphology::Type& bwNegOperation, const cv::Mat& bwNegImage, const int& structuringElement, const int& size);

    // Generate all combinations of erode, dilate operations with rect, cross & ellipse structuring elements.
    void ErodeDilateTestAll(const cv::Mat& image, const int& size);

    // Generate the window name for the Erode/Dilate test.
    std::string ErodeDilateWinName(const Morphology::Type& bwOperation, const Morphology::Type& bwNegOperation, const int& structuringElement);

    // Generate the window name for the Erode/Dilate test.
    std::string ErodeDilateWinName(const Morphology::Type& operation, const int& structuringElement, const bool& isNegImage);

    // Open an image: erode followed by dilate.
    void Open(const cv::Mat& image, const int& structuringElement, const int& size);

    // Close an image: dilate followed by erode.
    void Close(const cv::Mat& image, const int& structuringElement, const int& size);

    // Morphological gradient: dilated image - eroded image.
    void DoGradient(const cv::Mat& image, const Gradient::Type& type, const bool& doNegative, const int& structuringElement, const int& size);

    // Perform all morphological gradient combinations.
    void DoGradientAll(const cv::Mat& image, const int& structuringElement, const int& size);

    // calculate the absolute difference and use median blur to attenuate reflections
    cv::Mat Task03a(const cv::Mat& grayImage1, const cv::Mat& grayImage2, const std::string& folder, const std::string& winName);

    // filter noise & generate cup mask
    cv::Mat Task03b(const cv::Mat& blurImage, const std::string& folder, const std::string& winName);

    // flood fill test
    void Task03c(const cv::Mat& t03bImage, const std::string& folder, const std::string& winName);

    // perform the image composition using bitwise operations.
    void Task03e(const cv::Mat& srcImage1, const cv::Mat& srcImage2, const cv::Mat& t03bImage, const std::string& folder, const std::string& winName);
}

#endif