#ifndef LAB_04_H
#define LAB_04_H

#include "common.h"

#include <opencv2/features2d.hpp>

class Lab04
{
public:
    // The detectors used in lab 04 experiments.
    enum class Detector
    {
        ORB,
        SURF,
        NONE
    };

    // The matchers used in lab 04 experiments.
    enum class Matcher
    {
        BFmatcher,
        FlannBasedMatcher,
        NONE
    };

    // Types of filter supported by lab 04 experiments.
    enum class Filter
    {
        Median,
        Gaussian,
        Sharpen,
        NONE
    };

public:
    Lab04
        (
        const std::string& trnFile,
        const std::string& tstFile,
        const Detector& detectorType,
        const Matcher& matcherType
        );
    ~Lab04() { };

private:
    // remove default/copy constructor and assignment operator
    Lab04() = delete;
    Lab04(const Lab04& gen) = delete;
    Lab04(const Lab04 *const gen) = delete;
    Lab04(Lab04 *const gen) = delete;
    Lab04& operator=(const Lab04& gen) = delete;
    Lab04& operator=(const Lab04 *const gen) = delete;
    Lab04& operator=(Lab04 *const gen) = delete;

private:
    // training image
    cv::Mat trnImage;

    // test image
    cv::Mat tstImage;

    // the name of the training file without extension
    std::string trnName;

    // the name of the test file without extension
    std::string tstName;

    // this specifies the detector used
    Detector detectorType;

    // this specifies the matcher used
    Matcher matcherType;

public:
    // this specifies the filter filter used
    Filter filterType;

    // filter size (square)
    int ksize;
    // gaussian only
    double sigma;

    // total number of matches
    int nrMatchesTotal;

    // number of matches to keep after matching
    int nrMatches;

private:
    // training image key points
    std::vector<cv::KeyPoint> trnKeyPoints;

    // test image key points
    std::vector<cv::KeyPoint> tstKeyPoints;

    // training image descriptors
    cv::Mat trnDescriptors;

    // training image descriptors
    cv::Mat tstDescriptors;

public:
    // Experiment: Try to match key points between a test image and a training image.
    void Experiment();

    // Return the log containing the results of the matching operation.
    std::string Log();

private:
    // Detect key points and extract feature descriptors.
    void ExtractFeatures();

    // Match feature descriptors.
    void Match();

private:
    // Generate the output file name.
    std::string FileName(const std::string& name);

    // Convert Detector to string name.
    std::string ToString(const Detector& detectorType);

    // Convert Matcher to string name.
    std::string ToString(const Matcher& matcherType);

    // Convert filter to string name.
    std::string ToString(const Filter& filterType, const int& ksize, const double& sigma);
};

#endif