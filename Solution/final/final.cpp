#include <algorithm>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml.hpp>

#include "clock.h"
#include "final.h"

using namespace std;
using namespace cv;
using namespace ml;
using namespace common;

// synthetic test image - default dimensions
int srows = 240;
int scols = 320;

// ------------------------------------------------------------------------------------------------
// Main entry point.
// ------------------------------------------------------------------------------------------------
void main(int argc, char** argv)
{
    cout << "Embedded Vision I - Final Project - Dan Petre" << endl << endl;

    if (1 == argc)
    {
        CmdHelp();
        return;
    }

    bool displayDigits = false;
    bool displayPredictions = false;
    bool singleDigitTest = false;
    bool syntheticTest = false;
    bool suppress = false;

    // number of closest neighbors to check out
    int K = 1;

    // the number of randomly selected test samples
    int rndTestSize = 0;

    // the total number of digits in the synthetic test image
    int nrDigits = 0;

    // number of synthetic image tests to perform one after another
    int batchSize = 1;

    // read the cmd line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0) { CmdHelp(); return; }
        if (strcmp(argv[i], "-d") == 0) { displayDigits = true; continue; }
        if (strcmp(argv[i], "-p") == 0) { displayPredictions = true; continue; }
        if (strcmp(argv[i], "-k") == 0) { K = atoi(argv[++i]); continue; }
        if (strcmp(argv[i], "-r") == 0) { rndTestSize = atoi(argv[++i]); continue; }
        if (strcmp(argv[i], "-sd") == 0) { singleDigitTest = true; continue; }
        if (strcmp(argv[i], "-st") == 0) { syntheticTest = true; nrDigits = atoi(argv[++i]); continue; }
        if (strcmp(argv[i], "-cols") == 0) { scols = atoi(argv[++i]); continue; }
        if (strcmp(argv[i], "-rows") == 0) { srows = atoi(argv[++i]); continue; }
        if (strcmp(argv[i], "-b") == 0)
        {
            batchSize = atoi(argv[++i]);
            suppress = true;
            displayPredictions = false;
            continue;
        }
        if (strcmp(argv[i], "-s") == 0) { suppress = true; displayPredictions = false; continue; }
    }

    // the MNIST training and test data & labels sets
    string trnLabelFile = "c:/OpenCV.3.1/Solution/images/mnist/train-labels.idx1-ubyte";
    string trnImageFile = "c:/OpenCV.3.1/Solution/images/mnist/train-images.idx3-ubyte";
    string tstLabelFile = "c:/OpenCV.3.1/Solution/images/mnist/t10k-labels.idx1-ubyte";
    string tstImageFile = "c:/OpenCV.3.1/Solution/images/mnist/t10k-images.idx3-ubyte";

    // read the training images and labels
    idx1 mnistTrnLabels(trnLabelFile);
    idx3 mnistTrnImages(trnImageFile);

    // read the test images and labels
    idx1 mnistTstLabels(tstLabelFile);
    idx3 mnistTstImages(tstImageFile);

    if (displayDigits)
    {
        DisplayDigits(mnistTrnImages, mnistTrnLabels, mnistTrnImages.Rows(), mnistTrnImages.Cols());
        return;
    }

    if (singleDigitTest)
    {
        SingleDigitTest(mnistTrnImages, mnistTrnLabels, mnistTstImages, mnistTstLabels, K, rndTestSize, displayPredictions);
        return;
    }

    if (syntheticTest)
    {
        SyntheticTest(mnistTrnImages, mnistTrnLabels, mnistTstImages, mnistTstLabels, K, nrDigits, batchSize, displayPredictions, suppress);
        return;
    }

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Init rand engine.
// ------------------------------------------------------------------------------------------------
void Seed()
{
    //*///
    srand((unsigned int)time(NULL));
    /*/// for debug purposes - a way to log the seed
    unsigned int seed = ((unsigned int)time(NULL));
    srand(seed);
    cout << "Seed: " << seed << endl;
    //*///
}

// ------------------------------------------------------------------------------------------------
// Sorting rule for cv::Point. Sort by columns than by rows.
// ------------------------------------------------------------------------------------------------
struct
{
    bool operator() (cv::Point p1, cv::Point p2) { return ((p1.y * scols + p1.x) < (p2.y * scols + p2.x)); }
} PointRule;

// ------------------------------------------------------------------------------------------------
// Sorting rule for cv::Rect. Sort by columns than by rows.
// ------------------------------------------------------------------------------------------------
struct
{
    bool operator() (cv::Rect r1, cv::Rect r2) { return ((r1.y * scols + r1.x) < (r2.y * scols + r2.x)); }
} RectRule;

// ------------------------------------------------------------------------------------------------
// Sorting rule for cv::Rect. Sort by size.
// ------------------------------------------------------------------------------------------------
struct
{
    bool operator() (cv::Rect r1, cv::Rect r2) { return r1.area() > r2.area(); }
} RectRuleA;

// ------------------------------------------------------------------------------------------------
// Command line help.
// ------------------------------------------------------------------------------------------------
void CmdHelp()
{
    cout << "-h       ... display this help.                                                             " << endl;
    cout << "-d       ... display MNIST digits.                                          default: false  " << endl;
    cout << "-sd      ... perform the single digit detection test on all samples.                        " << endl;
    cout << "-r     N ... constrain the single digit detection test to N randomly selected test samples. " << endl;
    cout << "-st    N ... perform the detection over a synthetic image with N digits.                    " << endl;
    cout << "-k     N ... set the number of K nearest neighbors.                         default: 1      " << endl;
    cout << "-cols  C ... set the width (nr. columns) of the synthetic test image.       default: 320    " << endl;
    cout << "-rows  R ... set the height (nr. rows) of the synthetic test image.         default: 240    " << endl;
    cout << "-p       ... display predicted values.                                      default: false  " << endl;
    cout << "-b     N ... batch mode execution on synthetic test images.                                 " << endl;
    cout << "-s       ... suppress user interaction.                                                     " << endl;
    cout << endl;
}

// ------------------------------------------------------------------------------------------------
// Get a random integer value between 0 and vmax.
// ------------------------------------------------------------------------------------------------
int32_t rand(const int32_t& vmax, const int32_t& vmin)
{
    return (int32_t)(vmin + (vmax - vmin) * (int32_t)std::rand() / RAND_MAX);
}

// ------------------------------------------------------------------------------------------------
// Display a digit in its own window.
// ------------------------------------------------------------------------------------------------
KEY ShowDigit(const string& winName, const idx3& mnistTrnImages, const int& index)
{
    int rows = mnistTrnImages.Rows();
    int cols = mnistTrnImages.Cols();

    Mat digit(rows, cols, CV_32FC1, (void*)(mnistTrnImages[index]));

    int key = Show(digit, winName, (int)(cols * SCALE_FACTOR), (int)(rows * SCALE_FACTOR), CV_WINDOW_NORMAL, POS_X, POS_Y);

    destroyAllWindows();

    switch (key)
    {
    case (int)KEY::ESC: return KEY::ESC;
    default: return KEY::ELSE;
    }
}

// ------------------------------------------------------------------------------------------------
// Display a digit in its own window.
// ------------------------------------------------------------------------------------------------
KEY ShowDigit(const string& winName, const Mat& images, const int& rows, const int& cols, const int& index)
{
    Mat digit(rows, cols, CV_32FC1, (void*)(images.row(index).data));

    int key = Show(digit, winName, (int)(cols * SCALE_FACTOR), (int)(rows * SCALE_FACTOR), CV_WINDOW_NORMAL, POS_X, POS_Y);

    destroyAllWindows();

    switch (key)
    {
    case (int)KEY::ESC: return KEY::ESC;
    default: return KEY::ELSE;
    }
}

// ------------------------------------------------------------------------------------------------
// Display randomly picked digits, one by one, until the Esc key is pressed.
// ------------------------------------------------------------------------------------------------
void DisplayDigits
(
    const idx3& mnistImages,
    const idx1& mnistLabels,
    const int& rows,
    const int& cols
)
{
    Mat images = GetData(mnistImages);
    Mat labels = GetData(mnistLabels);

    Seed();

    while (true)
    {
        int32_t index = rand(images.rows);

        stringstream winName;
        winName << "Digit: " << labels.at<float>(index, 0);

        KEY key = ShowDigit(winName.str(), images, rows, cols, index);

        if (key == KEY::ESC)
            break;
    }
}

// ------------------------------------------------------------------------------------------------
// Get MNIST images as single cv::Mat. Each row is one MNIST image.
// ------------------------------------------------------------------------------------------------
Mat GetData(const idx3& images)
{
    int rows = images.Rows();
    int cols = images.Cols();

    int nrImages = images.Count();

    int imgSize = rows * cols;

    return Mat(nrImages, imgSize, CV_32FC1, (void*)images.ptr());
}

// ------------------------------------------------------------------------------------------------
// Get MNIST labels as single cv::Mat. Each row is one MNIST label.
// ------------------------------------------------------------------------------------------------
Mat GetData(const idx1& labels)
{
    int nrLabels = labels.Count();

    int lblSize = 1;

    return Mat(nrLabels, lblSize, CV_32FC1, (void*)labels.ptr());
}

// ------------------------------------------------------------------------------------------------
// Read the test data from the disk.
// ------------------------------------------------------------------------------------------------
void InitTestData
(
    const idx3& mnistTstImages,
    const idx1& mnistTstLabels,
    Mat& tstImages,
    Mat& tstLabels,
    const int& rndTestSize
)
{
    int rows = mnistTstImages.Rows();
    int cols = mnistTstImages.Cols();

    int tstCount = min(mnistTstImages.Count(), mnistTstLabels.Count());

    int imgSize = rows * cols;
    int lblSize = 1;

    if (0 == rndTestSize)
    {
        // initialize the test data
        tstImages = Mat(tstCount, imgSize, CV_32FC1, (void*)mnistTstImages.ptr()).clone();
        tstLabels = Mat(tstCount, lblSize, CV_32FC1, (void*)mnistTstLabels.ptr()).clone();
    }
    else
    {
        vector<float> rndImages(rndTestSize * imgSize);
        vector<float> rndLabels(rndTestSize * lblSize);

        Seed();

        // randomly select digits and their corresponding labels from the test set
        for (int i = 0; i < rndTestSize; i++)
        {
            int index = rand(tstCount);

            memcpy(&rndImages[i * imgSize], mnistTstImages[index], imgSize * sizeof(float));
            rndLabels[i] = mnistTstLabels[index];
        }

        tstImages = Mat(rndTestSize, imgSize, CV_32FC1, (void*)&rndImages[0]).clone();
        tstLabels = Mat(rndTestSize, lblSize, CV_32FC1, (void*)&rndLabels[0]).clone();
    }
}

// ------------------------------------------------------------------------------------------------
// Perform the single digit detection test.
// ------------------------------------------------------------------------------------------------
void SingleDigitTest
(
    const idx3& mnistTrnImages,
    const idx1& mnistTrnLabels,
    const idx3& mnistTstImages,
    const idx1& mnistTstLabels,
    const int& K,
    const int& testSize,
    const bool displayPredictions
)
{
    // training data
    Mat trnImages = GetData(mnistTrnImages);
    Mat trnLabels = GetData(mnistTrnLabels);

    // test data
    Mat tstImages;
    Mat tstLabels;

    InitTestData(mnistTstImages, mnistTstLabels, tstImages, tstLabels, testSize);

    KNN knn(trnImages, trnLabels, K);

    clock::start();

    knn.Predict(tstImages, tstLabels);

    clock::stop();

    double duration = clock::delta();

    DisplayStatistics(knn.Accuracy(), tstImages.rows, K, duration);

    if (!displayPredictions)
        return;

    DisplayPredictions(knn, tstImages.rows);
}

// ------------------------------------------------------------------------------------------------
// Perform the digit detection in a synthetic image test.
// ------------------------------------------------------------------------------------------------
void SyntheticTest
(
    const idx3& mnistTrnImages,
    const idx1& mnistTrnLabels,
    const idx3& mnistTstImages,
    const idx1& mnistTstLabels,
    const int& K,
    const int& testSize,
    const int& batchSize,
    const bool& displayPredictions,
    const bool& suppress
)
{
    // no digits - no test
    if (0 == testSize)
    {
        cout << "Please specify the number of digits." << endl;
        return;
    }

    stringstream winName;
    winName << "Synthetic test: " << scols << "x" << srows << " with " << testSize << " digits";

    // training data
    Mat trnImages = GetData(mnistTrnImages);
    Mat trnLabels = GetData(mnistTrnLabels);

    // train
    KNN knn(trnImages, trnLabels, K);

    Mat tstImages;
    Mat tstLabels;

    int tstCount = mnistTstImages.Count();
    int rows = mnistTstImages.Rows();
    int cols = mnistTstImages.Cols();

    int imgSize = rows * cols;
    int lblSize = 1;

    Scalar black(0.0, 0.0, 0.0);

    // create the synthetic image
    Mat synthetic;
    vector<float> rndLabels;

    // remember which digits where placed where
    vector<Point> rndLocations;

    vector<double> accs(batchSize);
    for (int b = 0; b < batchSize; b++)
    {
        synthetic = GetSyntheticImage(testSize, mnistTstLabels, mnistTstImages, rndLocations, rndLabels);

        if (!suppress)
            Show(synthetic, winName.str(), scols, srows);

        // get the bounding boxes around potential digits
        vector<Rect> boxes = FindBoundingBoxes(synthetic);

        if (!suppress)
            DisplayFoundDigits(synthetic, boxes, winName.str());

        RejectSmallBoxes(boxes, rows, cols);

        if (!suppress)
            DisplayFoundDigits(synthetic, boxes, winName.str());

        FuseOverlapping(boxes, rows, cols);

        Rect bounds(0, 0, synthetic.cols, synthetic.rows);

        ReSizeCenter(bounds, boxes, rows, cols);

        // sort the found bounding boxes (row major order)
        sort(boxes.begin(), boxes.end(), RectRule);

        tstLabels = GetTestLabels(boxes, rndLocations, rndLabels, rows, cols);

        if (!suppress)
            DisplayFoundDigits(synthetic, boxes, winName.str());

        tstImages = GetTestImages(synthetic, boxes, rows, cols);

        clock::start();

        knn.Predict(tstImages, tstLabels);

        clock::stop();

        double duration = clock::delta();
        double accuracy = knn.Accuracy();

        accs[b] = accuracy;

        DisplayStatistics(accuracy, tstImages.rows, K, duration, (b == 0) ? true : false);

        if (suppress)
            continue;

        waitKey();

        if (!displayPredictions)
            continue;

        DisplayPredictions(knn, tstImages.rows);

        waitKey();
    }

    // calculate statistics about the accuracy of the current method used for digit recognition
    if (100 <= batchSize)
    {
        double mean = 0.0;
        double sigma = 0.0;

        for (int b = 0; b < batchSize; b++)
            mean += accs[b];

        mean /= batchSize;

        for (int b = 0; b < batchSize; b++)
            sigma += (mean - accs[b]) * (mean - accs[b]);

        sigma /= batchSize;
        sigma = sqrt(sigma);

        cout << endl;
        cout << endl;
        cout << "Accuracy summary" << endl;
        cout << "mean:  " << setprecision(4) << mean  << "%" << endl;
        cout << "sigma: " << setprecision(4) << sigma << "%" << endl;
    }
}

// ------------------------------------------------------------------------------------------------
// Remove small boxes.
// ------------------------------------------------------------------------------------------------
void RejectSmallBoxes(vector<Rect>& boxes, const int& rows, const int& cols)
{
    vector<Rect> output;

    int rowsThreshold = (int)rows / 4;
    int colsThreshold = (int)cols / 4;

    for (size_t i = 0; i < boxes.size(); i++)
    {
        if ((boxes[i].width < colsThreshold) && (boxes[i].height < rowsThreshold))
            continue;

        output.push_back(boxes[i]);
    }

    boxes.swap(output);
}

// ------------------------------------------------------------------------------------------------
// Fuse overlapping boxes.
// ------------------------------------------------------------------------------------------------
void FuseOverlapping(vector<Rect>& boxes, const int& rows, const int& cols)
{
    // sort by size
    sort(boxes.begin(), boxes.end(), RectRuleA);

    vector<Rect> fused;
    vector<int> ignore;
    // fuse overlapping boxes
    for (size_t i = 0; i < boxes.size(); i++)
    {
        // skip the rectangles already fused
        if (ignore.end() != find(ignore.begin(), ignore.end(), i))
            continue;

        Rect box = boxes[i];
        for (size_t j = i + 1; j < boxes.size(); j++)
        {
            // skip the rectangles already fused
            if (ignore.end() != find(ignore.begin(), ignore.end(), j))
                continue;

            Rect crt = boxes[j];


            bool contains = false;
            // if these boxes are close enough it might, one might contain just noise
            if (box.area() >= crt.area())
            {
                Rect boxLarge(box.x - (cols - box.width) / 2, box.y - (rows - box.height) / 2, cols, rows);

                // check if the expanded box contains crt
                contains = ((boxLarge & crt).area() >= crt.area());
            }
            else
            {
                Rect crtLarge(crt.x - (cols - crt.width) / 2, crt.y - (rows - crt.height) / 2, cols, rows);

                // check if the expanded crt contains box
                contains = ((box & crtLarge).area() >= box.area());
            }

            if (!contains)
                continue;

            // fuse the rectangles
            box = box | crt;

            ignore.push_back(j);
            sort(ignore.begin(), ignore.end());
        }

        fused.push_back(box);
        ignore.push_back(i);
    }

    // put the fused results back in
    boxes.swap(fused);
}

// ------------------------------------------------------------------------------------------------
// Resize the boxes to match MNIST digit dimensions.
// Center the boxes around the original contents.
// ------------------------------------------------------------------------------------------------
void ReSizeCenter(const Rect& bounds, vector<Rect>& boxes, const int& rows, const int& cols)
{
    vector<Rect> output;

    // recenter-resize
    for (size_t i = 0; i < boxes.size(); i++)
    {
        // center the box inside the (cols, rows) box for a digit.
        int x = boxes[i].x - (cols - boxes[i].width) / 2;
        int y = boxes[i].y - (rows - boxes[i].height) / 2;

        // resize the box to the mnist digit dimensions
        int width = cols;
        int height = rows;

        if (!bounds.contains(Point(x, y)))
            continue;

        if (!bounds.contains(Point(x + width, y)))
            continue;

        if (!bounds.contains(Point(x, y + height)))
            continue;

        if (!bounds.contains(Point(x + width, y + height)))
            continue;

        output.push_back(Rect(x, y, width, height));
    }

    boxes.swap(output);
}

// ------------------------------------------------------------------------------------------------
// Generate an image with randomly placed digits. No overlapping.
// ------------------------------------------------------------------------------------------------
Mat GetSyntheticImage
(
    const int& nrDigits,
    const idx1& mnistTstLabels,
    const idx3& mnistTstImages,
    vector<Point>& rndLocations,
    vector<float>& rndLabels
)
{
    Scalar black(0.0, 0.0, 0.0);
    Mat synthetic(srows, scols, CV_32FC1, black);
    vector<Rect> rndBoxes;

    int tstCount = mnistTstImages.Count();
    int rows = mnistTstImages.Rows();
    int cols = mnistTstImages.Cols();

    Seed();

    for (int i = 0; i < nrDigits; i++)
    {
        int index = 0;
        int x = 0;
        int y = 0;

        // pick a random digit from the test set
        index = rand(tstCount - 1);

        bool done = false;
        do
        {
            // pick a random destination location
            x = rand(scols - (int)(1.5 * cols), (int)(0.5 * cols));
            y = rand(srows - (int)(1.5 * rows), (int)(0.5 * rows));

            Point location(x, y);
            Rect box(x, y, cols, rows);

            // first record
            if (0 == rndLocations.size())
            {
                // record the location and bounding box for algorithm evaluation purposes
                rndLocations.push_back(location);
                rndBoxes.push_back(box);

                break;
            }

            // make sure there's no bounding box overlap
            done = true;
            for (size_t j = 0; j < rndBoxes.size(); j++)
            {
                Rect crt = rndBoxes[j];
                bool intersect = ((box & crt).area() > 0);

                if (intersect)
                {
                    done = false;
                    break;
                }
            }

            if (done)
            {
                // record the location and bounding box for algorithm evaluation purposes
                rndLocations.push_back(location);
                rndBoxes.push_back(box);

                break;
            }
        } while (!done);

        // get the digit data
        Mat digit(rows, cols, CV_32FC1, (void*)(mnistTstImages[index]));

        // copy the digit over to the synthetic image
        digit.copyTo(synthetic(Rect(x, y, cols, rows)));

        rndLabels.push_back(mnistTstLabels[index]);
    }

    return synthetic;
}

// ------------------------------------------------------------------------------------------------
// Find bounding boxes around potential digits.
// ------------------------------------------------------------------------------------------------
vector<Rect> FindBoundingBoxes(const Mat& synthetic)
{
    // find bounding boxes around the randomly placed digits
    Mat contoursMat;
    vector<vector<Point>> contours;

    // find contours requires CV_8UC1
    contoursMat = synthetic.clone();
    contoursMat *= 255; // scale back to the full range of CV_8UC1
    contoursMat.convertTo(contoursMat, CV_8UC1);

    // Find contours
    findContours(contoursMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // get the bounding boxes around potential digits
    vector<Rect> boxes;
    boxes.reserve(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        Rect box = boundingRect(Mat(contours[i]));

        boxes.push_back(box);
    }

    return boxes;
}

// ------------------------------------------------------------------------------------------------
// Get the expected labels for the digits in the synthetic image.
// Knowing where the digits are located in the synthetic image we can establish the gold truth.
// ------------------------------------------------------------------------------------------------
Mat GetTestLabels
(
    const vector<Rect>& boxes,
    const vector<Point>& rndLocations,
    const vector<float>& rndLabels,
    const int& rows,
    const int& cols
)
{
    if (0 == boxes.size())
        return Mat();

    vector<float> digits(boxes.size(), -1);
    for (size_t i = 0; i < boxes.size(); i++)
    {
        Rect box = boxes[i];
        for (size_t j = 0; j < rndLocations.size(); j++)
        {
            Rect ref(rndLocations[j].x, rndLocations[j].y, cols, rows);

            // "same" box if it overlaps 80%
            int areaThreshold = (int)(0.8 * ref.area());

            // check if the boxes overlap
            bool intersect = (box & ref).area() > areaThreshold;

            if (intersect)
            {
                digits[i] = rndLabels[j];
            }
        }
    }

    return Mat(boxes.size(), 1, CV_32FC1, (void*)&digits[0]).clone();
}

// ------------------------------------------------------------------------------------------------
// Display the potential digits identified in the synthetic image.
// ------------------------------------------------------------------------------------------------
void DisplayFoundDigits(const Mat& synthetic, const vector<Rect>& boxes, const string& winName)
{
    // original image is CV_32FC1 with values between 0.0 and 1.0
    Mat found = synthetic.clone();

    // scale to values between 0-255
    found *= 255;

    // convert to 3 channel
    found.convertTo(found, CV_8UC1);
    cvtColor(found, found, CV_GRAY2RGB);

    Scalar green(0.0, 255.0, 0.0);
    int thickness = 1;

    for (size_t i = 0; i < boxes.size(); i++)
    {
        Point p1 = Point(boxes[i].x, boxes[i].y);
        Point p2 = Point(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);

        // draw the bounding box
        rectangle(found, p1, p2, green, thickness);
    }

    Show(found, winName, scols, srows);
}

// ------------------------------------------------------------------------------------------------
// Get the test images in the format expected by the cv::KNearest class.
// ------------------------------------------------------------------------------------------------
Mat GetTestImages
(
    const Mat& synthetic,
    const vector<Rect>& boxes,
    const int& rows,
    const int& cols
)
{
    if (0 == boxes.size())
        return Mat();

    int imgSize = rows * cols;
    Scalar black(0.0, 0.0, 0.0);

    // temporary array that helps in the extraction of the test fragments
    vector<float> rndImages(boxes.size() * imgSize);

    for (size_t i = 0; i < boxes.size(); i++)
    {
        Mat roi = synthetic(Rect(boxes[i].x, boxes[i].y, cols, rows));

        // use thresholding to discard empty fragments
        if (black == sum(roi))
            continue;

        // copy the fragment to the test image
        for (int j = 0; j < rows; j++)
        {
            float* roiPtr = roi.row(j).ptr<float>(0);
            memcpy(&rndImages[i * rows * cols + j * cols], roiPtr, cols * sizeof(float));
        }
    }

    return Mat(boxes.size(), imgSize, CV_32FC1, (void*)&rndImages[0]).clone();
}

// ------------------------------------------------------------------------------------------------
// Display test statistics.
// ------------------------------------------------------------------------------------------------
void DisplayStatistics
(
    const double& accuracy,
    const int& testSize,
    const int& K,
    const double& duration,
    const bool& header
)
{
    if (header)
    {
        cout << left << setw(16) << "Accuracy (%):  ";
        cout << left << setw(16) << "Test Size:     ";
        cout << left << setw(16) << "Value of K:    ";
        cout << left << setw(16) << "Duration (ms): ";
        cout << left << setw(16) << "FPS:           ";
        cout << endl;
    }

    cout << left << setw(16) << accuracy;
    cout << left << setw(16) << testSize;
    cout << left << setw(16) << K;
    cout << left << setw(16) << round(duration);
    cout << left << setw(16) << round(testSize / duration * 1000);

    cout << endl;
}

// ------------------------------------------------------------------------------------------------
// Display predicted values side by side with expected/true values.
// ------------------------------------------------------------------------------------------------
void DisplayPredictions(const KNN& knn, const int& testSize)
{
    cout << "True value : Predicted value" << endl;
    int width = (int)sqrt(testSize);
    for (int i = 0; i < testSize; i++)
    {
        Pair pair = knn[i];

        cout << setw(2) << pair.trueValue;
        cout << ":";
        cout << setw(2) << pair.predicted;
        cout << "\t";

        if (0 == ((i + 1) % width))
            cout << endl;
    }
    cout << endl;
}
