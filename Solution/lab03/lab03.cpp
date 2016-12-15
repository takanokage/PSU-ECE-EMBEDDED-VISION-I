#include <iostream>
#include <iomanip>
#include <string>

#include "lab03.h"
#include "histogram.h"

#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace common;

const string LAB_03_PATH = "C:\\OpenCV.3.1\\Solution\\images\\lab03\\";
const string TASK_01_PATH = LAB_03_PATH + "task01\\";
const string TASK_02_PATH = LAB_03_PATH + "task02\\";
const string TASK_03_PATH = LAB_03_PATH + "task03\\";

// full path to the root folder of the executable
string root = "";

Mat image;
int counter = 0;

void main(int argc, char** argv)
{
    cout << "Lab03 - Dan Petre" << endl << endl;

    root = FileFolder(argv[0]);

    bool captureFaces = false;

    if (argc < 3)
    {
        cout << "-g <image_path>" << " ... " << "image showing a group of people." << endl;
        cout << "-t <image_path>" << " ... " << "face template." << endl;
        cout << "-c             " << " ... " << "enable face capture on mouse left-click." << endl;
        cout << endl;
    }

    string groupFile = "";
    string templateFile = "";
    string t03Image = "";

    // read the cmd line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-g") == 0) { groupFile = argv[++i]; }
        if (strcmp(argv[i], "-t") == 0) { templateFile = argv[++i]; }
        if (strcmp(argv[i], "-c") == 0) { captureFaces = true; }
    }

    // make sure the files exist
    if (!Exists(groupFile)) { cout << "Please provide a valid group image." << endl; return; }
    if (!Exists(templateFile)) { cout << "Please provide a valid face template image." << endl; return; }

    if (captureFaces)
        lab03::CaptureFaces(groupFile);

    lab03::Task01(groupFile, templateFile);
    lab03::Task01b(groupFile, templateFile);
    lab03::Task02(groupFile, templateFile);
    lab03::Task03(groupFile);
}

// ------------------------------------------------------------------------------------------------
// Task 01: Histogram based face detection.
// ------------------------------------------------------------------------------------------------
void lab03::Task01(const string& groupFile, const string& templateFile)
{
    string winName = "Task 01: Histogram based face detection.";
    int tmpScale = 5;

    Size gsize(3, 3);
    double sigmaX = 0.9;

    Mat grpImage = imread(groupFile);
    //medianBlur(grpImage, grpImage, 3);
    //GaussianBlur(grpImage, grpImage, gsize, sigmaX);
    Show(grpImage, winName, grpImage.cols, grpImage.rows);

    Mat tmpImage = imread(templateFile);
    //medianBlur(tmpImage, tmpImage, 3);
    //GaussianBlur(tmpImage, tmpImage, gsize, sigmaX);
    Show(tmpImage, winName, tmpScale * tmpImage.cols, tmpScale * tmpImage.rows);

    // reduce colors
    Mat grpImageRed = ColorReduce(grpImage);
    Show(grpImageRed, winName, grpImageRed.cols, grpImageRed.rows);
    Save(grpImageRed, TASK_01_PATH + "grpImageRed.jpg");

    Mat tmpImageRed = ColorReduce(tmpImage);
    Show(tmpImageRed, winName, tmpScale * tmpImageRed.cols, tmpScale * tmpImageRed.rows);
    Save(tmpImageRed, TASK_01_PATH + "tmpImageRed.jpg");

    // obtain the histograms
    Histogram tmpHist(&tmpImageRed);

    float range[2] = { 0, 255 };
    const float* ranges[3] = { range, range, range };
    int channels[3] = { 0, 1, 2 };

    // The back projection of the group image
    Mat bckProj = tmpHist.BackProjection(grpImageRed);
    Show(bckProj, winName, bckProj.cols, bckProj.rows);
    Save(bckProj, TASK_01_PATH + "bckProj.jpg");

    // Threshold back projection to obtain a binary image
    double threshold = 0.4;
    cv::threshold(bckProj, bckProj, 255 * threshold, 255, cv::THRESH_BINARY);
    Show(bckProj, winName, bckProj.cols, bckProj.rows);
    Save(bckProj, TASK_01_PATH + "bckProj.thr.jpg");

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Task 01: attempt to improve the algorithm.
// ------------------------------------------------------------------------------------------------
void lab03::Task01b(const string& groupFile, const string& templateFile)
{
    string winName = "Task 01: attempt to improve the algorithm.";
    int tmpScale = 5;

    Size gsize(3, 3);
    double sigmaX = 0.9;

    Mat grpImage = imread(groupFile);
    Show(grpImage, winName, grpImage.cols, grpImage.rows);

    Mat tmpImage = imread(templateFile);
    medianBlur(tmpImage, tmpImage, 3);
    Show(tmpImage, winName, tmpScale * tmpImage.cols, tmpScale * tmpImage.rows);

    // reduce colors
    Mat grpImageRed = ColorReduce(grpImage);
    Show(grpImageRed, winName, grpImageRed.cols, grpImageRed.rows);
    Save(grpImageRed, TASK_01_PATH + "grpImageRed.jpg");

    Mat tmpImageRed = ColorReduce(tmpImage);
    Show(tmpImageRed, winName, tmpScale * tmpImageRed.cols, tmpScale * tmpImageRed.rows);
    Save(tmpImageRed, TASK_01_PATH + "tmpImageRed.jpg");

    // obtain the histograms
    Histogram tmpHist(&tmpImageRed);

    float range[2] = { 0, 255 };
    const float* ranges[3] = { range, range, range };
    int channels[3] = { 0, 1, 2 };

    // The back projection of the group image
    Mat bckProj = tmpHist.BackProjection(grpImageRed);
    Show(bckProj, winName, bckProj.cols, bckProj.rows);
    Save(bckProj, TASK_01_PATH + "bckProj.jpg");

    // Threshold back projection to obtain a binary image
    double threshold = 0.4;
    cv::threshold(bckProj, bckProj, 255 * threshold, 255, cv::THRESH_BINARY);
    Show(bckProj, winName, bckProj.cols, bckProj.rows);
    Save(bckProj, TASK_01_PATH + "bckProj.thr.jpg");

    // The eroded back projection
    Mat filtered = bckProj.clone();

    // blur the threshold image with a large kernel
    medianBlur(filtered, filtered, 15);
    Show(filtered, winName, filtered.cols, filtered.rows);

    // build the morphological kernel
    int strElem = MORPH_ELLIPSE;
    int size = 9;
    Size ksize = Size(size, size);
    Point anchor = Point((size - 1) / 2, (size - 1) / 2);
    Mat kernel = getStructuringElement(strElem, ksize, anchor);

    // perform open once to remove additional noise
    erode(filtered, filtered, kernel, anchor);
    Show(filtered, winName, filtered.cols, filtered.rows);

    dilate(filtered, filtered, kernel, anchor);
    Show(filtered, winName, filtered.cols, filtered.rows);

    // dilate 2 times to obtain large blobs
    dilate(filtered, filtered, kernel, anchor);
    Show(filtered, winName, filtered.cols, filtered.rows);

    dilate(filtered, filtered, kernel, anchor);
    Show(filtered, winName, filtered.cols, filtered.rows);

    Save(filtered, TASK_01_PATH + "bckProj.blobs.jpg");

    // perform masking on the original group image in order to assist in the evaluation of the algorithm
    Mat mask;
    cvtColor(filtered, mask, CV_GRAY2RGB);

    Mat masked;
    bitwise_and(grpImage, mask, masked);
    Show(masked, winName, masked.cols, masked.rows);
    Save(masked, TASK_01_PATH + "masked.jpg");

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Task 02: Template matching.
// ------------------------------------------------------------------------------------------------
void lab03::Task02(const string& groupFile, const string& templateFile)
{
    string winName = "Task 02: Template matching.";

    int tmpScale = 5;

    Size gsize(3, 3);
    double sigmaX = 0.9;

    Mat grpImage = imread(groupFile);
    //medianBlur(grpImage, grpImage, 3);
    //GaussianBlur(grpImage, grpImage, gsize, sigmaX);
    Show(grpImage, winName, grpImage.cols, grpImage.rows);

    Mat tmpImage = imread(templateFile);
    //medianBlur(tmpImage, tmpImage, 3);
    //GaussianBlur(tmpImage, tmpImage, gsize, sigmaX);
    Show(tmpImage, winName, tmpScale * tmpImage.cols, tmpScale * tmpImage.rows);

    // convert input images to gray
    Mat grpGray;
    cvtColor(grpImage, grpGray, CV_RGB2GRAY);
    Show(grpGray, winName, grpGray.cols, grpGray.rows);

    Mat tmpGray;
    cvtColor(tmpImage, tmpGray, CV_RGB2GRAY);
    Show(tmpGray, winName, tmpScale * tmpGray.cols, tmpScale * tmpGray.rows);

    // normalized similarity method
    int method = CV_TM_CCORR_NORMED;

    // perform matchTemplate and normalize the result
    Mat matched;
    matchTemplate(grpGray, tmpGray, matched, method);
    normalize(matched, matched, 0, 1, NORM_MINMAX);
    Show(matched, winName, matched.cols, matched.rows);

    // search for all matches above a threshold
    double threshold = 0.93;
    int strideX = 1;// tmpImage.cols / 4;
    int strideY = 1;// tmpImage.rows / 4;
    vector<Point> points;
    float* data = reinterpret_cast<float*>(matched.data);
    for (int row = 0; row < matched.rows; row += strideY)
    {
        for (int col = 0; col < matched.cols; col += strideX)
        {
            if (data[row * matched.cols + col] >= threshold)
            {
                Point point(col, row);

                points.push_back(point);
            }
        }
    }

    // draw green boxes around the matches and show/save the result
    Mat found = grpImage.clone();
    Scalar green(0.0, 255.0, 0.0);
    int thickness = 1;
    int boxWidth = 24;
    int boxHeight = 30;
    for (size_t i = 0; i < points.size(); i++)
    {
        Point p1 = Point(points[i].x - boxWidth / 2, points[i].y - boxHeight / 2);
        Point p2 = Point(points[i].x + boxWidth, points[i].y + boxHeight);

        rectangle(found, p1, p2, green, thickness);
    }
    Show(found, winName, found.cols, found.rows);
    Save(found, TASK_02_PATH + "found.jpg");

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Task 03: Find contours.
// ------------------------------------------------------------------------------------------------
void lab03::Task03(const string& imageFile)
{
    string winName = "Task 03: Find contours.";

    Mat originalImage = imread(imageFile);
    Show(originalImage, winName, originalImage.cols, originalImage.rows);

    // blur
    Mat blurryImage;
    Size gsize(3, 3);
    double sigmaX = 0.9;
    //GaussianBlur(originalImage, blurryImage, gsize, sigmaX);
    medianBlur(originalImage, blurryImage, 9);
    Show(blurryImage, winName, blurryImage.cols, blurryImage.rows);

    // Canny
    double threshold1 = 100;
    double threshold2 = 200;

    Mat cannyEdges;
    Canny(blurryImage, cannyEdges, threshold1, threshold2);
    Show(cannyEdges, winName, cannyEdges.cols, cannyEdges.rows);
    Save(cannyEdges, TASK_03_PATH + "cannyEdges.jpg");

    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    int mode = CV_RETR_TREE;
    int method = CV_CHAIN_APPROX_SIMPLE;

    findContours(cannyEdges, contours, hierarchy, mode, method, Point(0, 0));

    Mat found(originalImage.size(), originalImage.type(), Scalar());
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color(0.0, 255.0, 0.0);
        drawContours(found, contours, i, color);
    }
    Show(found, winName, found.cols, found.rows);
    Save(found, TASK_03_PATH + "findContours.jpg");

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Extract an image fragment and save it to disk.
// The image is saved to <solution>\images\lab03\task01.
// The image file name is face###.jpg where ### is a global counter initilized to zero at program start.
// The ROI is 20 rows x 16 cols and it is centered on the mouse cursor.
// ------------------------------------------------------------------------------------------------
void lab03::ExtractROI(int event, int x, int y, int flags, void* userdata)
{
    switch (event)
    {
    case EVENT_LBUTTONDOWN:
    {
        Rect box = Rect(x - 8, y - 10, 16, 20);

        Mat roi(image, box);

        stringstream imageFile;

        imageFile << TASK_01_PATH;
        imageFile << "face";
        imageFile << setw(3) << setfill('0') << counter;
        imageFile << ".jpg";

        Save(roi, imageFile.str());
        counter++;

        break;
    }
    case EVENT_RBUTTONDOWN: break;
    case EVENT_MBUTTONDOWN: break;
    case EVENT_MOUSEMOVE:   break;
    default: break;
    }
}

// ------------------------------------------------------------------------------------------------
// Face capture on mouse left button click. 
// ------------------------------------------------------------------------------------------------
void lab03::CaptureFaces(const string& imageFile)
{
    string winName = "Capture faces.";

    //Mat srcImage = ReadImage(imageFile, winName, true);

    image = imread(imageFile, 1);

    // show the input image
    namedWindow(winName, CV_WINDOW_NORMAL);

    //set the mouse callback function that does the actual capture.
    setMouseCallback(winName, ExtractROI, NULL);

    imshow(winName, image);
    resizeWindow(winName, image.cols, image.rows);

    waitKey();

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Color reduce an image.
// ------------------------------------------------------------------------------------------------
Mat lab03::ColorReduce(const Mat & image, const int & div)
{
    // log base2 of div = number of digits necessary to represent div in base2 
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));

    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

    cv::Mat_<cv::Vec3b>::const_iterator it = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::const_iterator itend = image.end<cv::Vec3b>();

    // Set output image (always 1-channel)
    cv::Mat result(image.rows, image.cols, image.type());
    cv::Mat_<cv::Vec3b>::iterator itr = result.begin<cv::Vec3b>();

    for (; it != itend; ++it, ++itr)
    {
        (*itr)[0] = ((*it)[0] & mask) + div / 2;
        (*itr)[1] = ((*it)[1] & mask) + div / 2;
        (*itr)[2] = ((*it)[2] & mask) + div / 2;
    }

    return result;
}
