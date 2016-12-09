#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "Lab04.h"

using namespace std;
using namespace cv;
using namespace common;

// Lab 04 windows name.
const string WIN_NAME = "Lab 04: Key point detection and matching.";

// Output path when saving images to disk.
const string LAB_04_OUTPUT = "C:\\OpenCV\\EBDVI_dpetre\\images\\Lab04\\experiments\\";

// Scale factors used to fit images to the screen.
const double SCALE_FACTOR = 0.75;

void main(int argc, char** argv)
{
    cout << "Lab04 - Dan Petre" << endl << endl;

    bool captureFaces = false;

    if (argc < 3)
    {
        cout << "-trn <image_path>" << " ... " << "the training image." << endl;
        cout << "-tst <image_path>" << " ... " << "the test image." << endl;
        cout << "-orb             " << " ... " << "choose the ORB detector." << endl;
        cout << "-surf            " << " ... " << "choose the SURF detector." << endl;
        cout << "-bf              " << " ... " << "choose the brute-force matcher." << endl;
        cout << "-flann           " << " ... " << "choose the Flann-based matcher." << endl;
        cout << "-m K             " << " ... " << "perform a median filter with a kernel size of K on the input images." << endl;
        cout << "-g K S           " << " ... " << "perform a Gaussian filter with a kernel size of K and sigma S on the input images." << endl;
        cout << "-s K S           " << " ... " << "sharpen the image using a Gaussian filter with a kernel size of K and sigma S on the input images." << endl;
        cout << "-n N             " << " ... " << "number of matches to keep & display. default 25." << endl;
        cout << "-noOut           " << " ... " << "disable waiting for a key press after each image." << endl;
        cout << endl;
    }

    string trnFile = "";
    string tstFile = "";

    Lab04::Detector detectorType = Lab04::Detector::NONE;
    Lab04::Matcher matcherType = Lab04::Matcher::NONE;
    Lab04::Filter filterType = Lab04::Filter::NONE;
    int ksize = 3;
    double sigma = 1.0;
    int nrMatches = 25;

    // read the cmd line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-trn") == 0) { trnFile = argv[++i]; }
        if (strcmp(argv[i], "-tst") == 0) { tstFile = argv[++i]; }
        if (strcmp(argv[i], "-orb") == 0) { detectorType = Lab04::Detector::ORB; }
        if (strcmp(argv[i], "-surf") == 0) { detectorType = Lab04::Detector::SURF; }
        if (strcmp(argv[i], "-bf") == 0) { matcherType = Lab04::Matcher::BFmatcher; }
        if (strcmp(argv[i], "-flann") == 0) { matcherType = Lab04::Matcher::FlannBasedMatcher; }
        if (strcmp(argv[i], "-m") == 0)
        {
            filterType = Lab04::Filter::Median;
            ksize = atoi(argv[++i]);
        }
        if (strcmp(argv[i], "-g") == 0)
        {
            filterType = Lab04::Filter::Gaussian;
            ksize = atoi(argv[++i]);
            sigma = atof(argv[++i]);
        }
        if (strcmp(argv[i], "-s") == 0)
        {
            filterType = Lab04::Filter::Sharpen;
            ksize = atoi(argv[++i]);
            sigma = atof(argv[++i]);
        }
        if (strcmp(argv[i], "-n") == 0) { nrMatches = atoi(argv[++i]); }
        if (strcmp(argv[i], "-noOut") == 0) { common::output = false; }
    }

    // cmd line arguments sanity check
    bool validArgs = true;
    if (!Exists(trnFile)) { cout << "Please provide a valid training image." << endl; validArgs = false; }
    if (!Exists(tstFile)) { cout << "Please provide a valid test image." << endl; validArgs = false; }
    if (Lab04::Detector::NONE == detectorType) { cout << "Please choose a detector." << endl; validArgs = false; }
    if (Lab04::Matcher::NONE == matcherType) { cout << "Please choose a matcher." << endl; validArgs = false; }

    if (validArgs)
    {
        Lab04 lab04(trnFile, tstFile, detectorType, matcherType);

        lab04.filterType = filterType;
        lab04.ksize = ksize;
        lab04.sigma = sigma;
        lab04.nrMatches = nrMatches;

        lab04.Experiment();

        cout << lab04.Log();
    }
}

// ------------------------------------------------------------------------------------------------
// Initialization constructor.
// ------------------------------------------------------------------------------------------------
Lab04::Lab04
(
const std::string& trnFile,
const std::string& tstFile,
const Detector& detectorType,
const Matcher& matcherType
) : detectorType(detectorType), matcherType(matcherType), filterType(Filter::NONE), ksize(3), sigma(1.0), nrMatches(25)
{
    trnImage = imread(trnFile);
    tstImage = imread(tstFile);

    trnName = common::FileName(trnFile);
    tstName = common::FileName(tstFile);
}

// ------------------------------------------------------------------------------------------------
// Experiment: Try to match key points between a test image and a training image.
// ------------------------------------------------------------------------------------------------
void Lab04::Experiment()
{
    Show(trnImage, WIN_NAME, (int)(trnImage.cols * SCALE_FACTOR), (int)(trnImage.rows * SCALE_FACTOR));

    Show(tstImage, WIN_NAME, (int)(tstImage.cols * SCALE_FACTOR), (int)(tstImage.rows * SCALE_FACTOR));

    // apply filter on the input images on user request
    switch (filterType)
    {
    case Filter::Median:
    {
        medianBlur(trnImage, trnImage, ksize);
        medianBlur(tstImage, tstImage, ksize);

        break;
    }
    case Filter::Gaussian:
    {
        GaussianBlur(trnImage, trnImage, Size(ksize, ksize), sigma);
        GaussianBlur(tstImage, tstImage, Size(ksize, ksize), sigma);

        break;
    }
    case Filter::Sharpen:
    {
        Mat aux;
        double alpha = 1.5;
        double beta = -0.5;
        double gamma = 0.0;

        // subtract the blurred image from the original
        // use a weighted sum function
        GaussianBlur(trnImage, aux, Size(ksize, ksize), sigma);
        addWeighted(trnImage, alpha, aux, beta, gamma, trnImage);

        GaussianBlur(tstImage, aux, Size(ksize, ksize), sigma);
        addWeighted(tstImage, alpha, aux, beta, gamma, tstImage);

        break;
    }
    }

    ExtractFeatures();

    Match();
}

// ------------------------------------------------------------------------------------------------
// Return the log containing the results of the matching operation.
// ------------------------------------------------------------------------------------------------
string Lab04::Log()
{
    stringstream output;

    output << left << setw(30) << "nr. key points (trn)";
    output << left << setw(30) << "nr. key points (tst)";
    output << left << setw(30) << "nr. matches         ";
    output << left << setw(30) << "detector type       ";
    output << left << setw(30) << "matcher type        ";
    output << left << setw(30) << "filter type         ";
    output << endl;
    output << left << setw(30) << trnKeyPoints.size();
    output << left << setw(30) << tstKeyPoints.size();
    output << left << setw(30) << nrMatchesTotal;
    output << left << setw(30) << ToString(detectorType);
    output << left << setw(30) << ToString(matcherType);
    output << left << setw(30) << ToString(filterType, ksize, sigma);

    return output.str();
}

// ------------------------------------------------------------------------------------------------
// Detect key points and extract feature descriptors.
// ------------------------------------------------------------------------------------------------
void Lab04::ExtractFeatures()
{
    const int flags = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
    const Scalar COLOR(255, 255, 255);

    // generic detector
    FeatureDetector* detector = NULL;

    // generic descriptor
    DescriptorExtractor* descriptor = NULL;

    // pick the detector based on its type
    switch (detectorType)
    {
    case Detector::ORB:
    {
        detector = new OrbFeatureDetector();
        descriptor = new OrbDescriptorExtractor();

        break;
    }
    case Detector::SURF:
    {
        detector = new SurfFeatureDetector(3000);
        descriptor = new SurfDescriptorExtractor();

        break;
    }
    default: break;
    }

    // identify the key points
    detector->detect(trnImage, trnKeyPoints);
    detector->detect(tstImage, tstKeyPoints);

    // based on the key points compute the descriptors
    descriptor->compute(trnImage, trnKeyPoints, trnDescriptors);
    descriptor->compute(tstImage, tstKeyPoints, tstDescriptors);

    // the descriptors need to be converted to float 32b if they're produced by the OrbDescriptorExtractor
    // otherwise the application crashes when trying to match them with the FlannBasedMatcher
    switch (detectorType)
    {
    case Detector::ORB:
    {
        if (CV_32F != trnDescriptors.type())
            trnDescriptors.convertTo(trnDescriptors, CV_32F);

        if (CV_32F != tstDescriptors.type())
            tstDescriptors.convertTo(tstDescriptors, CV_32F);

        break;
    }
    default: break;
    }

    // draw the keypoints as smaller/larger circles based on their scale
    Mat kpImage;

    drawKeypoints(trnImage, trnKeyPoints, kpImage, COLOR, flags);
    Show(kpImage, WIN_NAME, (int)(kpImage.cols * SCALE_FACTOR), (int)(kpImage.rows * SCALE_FACTOR));
    Save(kpImage, LAB_04_OUTPUT + FileName("keyPoints.trn"));

    drawKeypoints(tstImage, tstKeyPoints, kpImage, COLOR, flags);
    Show(kpImage, WIN_NAME, (int)(kpImage.cols * SCALE_FACTOR), (int)(kpImage.rows * SCALE_FACTOR));
    Save(kpImage, LAB_04_OUTPUT + FileName("keyPoints.tst"));

    // cleanup
    delete detector;
    delete descriptor;

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Match feature descriptors.
// ------------------------------------------------------------------------------------------------
void Lab04::Match()
{
    const Scalar COLOR(255, 255, 255);

    // generic matcher
    DescriptorMatcher* matcher = NULL;

    // vector of matches
    vector<DMatch> matches;

    // pick the matcher based on its type
    switch (matcherType)
    {
    case Matcher::BFmatcher:
    {
        matcher = new BFMatcher(NORM_L2, false);

        break;
    }
    case Matcher::FlannBasedMatcher:
    {
        matcher = new FlannBasedMatcher();

        break;
    }
    default: break;
    }

    // perform the matching between the two descriptor sets
    matcher->match(trnDescriptors, tstDescriptors, matches);

    nrMatchesTotal = matches.size();

    // get only the first closest NR_MATCHES
    if ((size_t)nrMatches < matches.size())
    {
        nth_element(matches.begin(), matches.begin() + nrMatches, matches.end());
        matches.erase(matches.begin() + nrMatches, matches.end());
    }

    // draw the matches on the training & test images side by side
    Mat output;
    drawMatches
        (
        trnImage, trnKeyPoints,  // the training image and its keypoints
        tstImage, tstKeyPoints,  // the test image and its keypoints
        matches,
        output,
        COLOR
        );

    Show(output, WIN_NAME, (int)(output.cols * SCALE_FACTOR), (int)(output.rows * SCALE_FACTOR));
    Save(output, LAB_04_OUTPUT + FileName("matched"));

    // cleanup
    delete matcher;

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Generate the output file name.
// ------------------------------------------------------------------------------------------------
string Lab04::FileName(const string& name)
{
    stringstream output;

    output << tstName << ".";
    output << ToString(detectorType) << ".";
    output << ToString(matcherType) << ".";
    output << ToString(filterType, ksize, sigma) << ".";
    output << nrMatches << ".";
    output << name << ".jpg";

    return output.str();
}

// ------------------------------------------------------------------------------------------------
// Convert Detector to string name.
// ------------------------------------------------------------------------------------------------
std::string Lab04::ToString(const Detector& detectorType)
{
    switch (detectorType)
    {
    case Detector::ORB: return "ORB";
    case Detector::SURF: return "SURF";
    default: return "NONE";
    }
}

// ------------------------------------------------------------------------------------------------
// Convert Matcher to string name.
// ------------------------------------------------------------------------------------------------
std::string Lab04::ToString(const Matcher& matcherType)
{
    switch (matcherType)
    {
    case Matcher::BFmatcher: return "BF";
    case Matcher::FlannBasedMatcher: return "Flann";
    default: return "NONE";
    }
}

// ------------------------------------------------------------------------------------------------
// Convert filter to string name.
// ------------------------------------------------------------------------------------------------
std::string Lab04::ToString(const Filter& filterType, const int& ksize, const double& sigma)
{
    stringstream output;

    switch (filterType)
    {
    case Filter::Median: output << "Median." << ksize << "x" << ksize; break;
    case Filter::Gaussian: output << "Gaussian." << ksize << "x" << ksize << "." << sigma; break;
    case Filter::Sharpen: output << "Sharpen." << ksize << "x" << ksize << "." << sigma; break;
    default: output << "no-filter"; break;
    }

    return output.str();
}
