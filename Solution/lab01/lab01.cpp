#include <iostream>
#include <string>

#include "lab01.h"

#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace common;

void main(int argc, char** argv)
{
    cout << "Lab01 - Dan Petre" << endl << endl;

    if ((argc < 2) || !Exists(argv[1]))
    {
        cout << "Please provide the path to an image." << endl;

        return;
    }

    //lab01::Task02(argv[1]);
    lab01::Task03();

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Lab 01 image pyramid
// ------------------------------------------------------------------------------------------------
cv::Mat lab01::ImagePyramid(pyramid* func, const cv::Mat& image, const int& scaleSteps, const float& scaleFactor, const string& winName, const int& cols, const int& rows)
{
    int border = BORDER_REFLECT_101;

    Mat srcImage;
    Mat dstImage;

    srcImage = image.clone();

    for (int i = 0; i < scaleSteps; i++)
    {
        Size size((int)(srcImage.cols * scaleFactor), (int)(srcImage.rows * scaleFactor));

        // scale UP/DOWN
        (*func)(srcImage, dstImage, size, border);

        Show(dstImage, winName, cols, rows);

        srcImage = dstImage.clone();
    }

    return dstImage;
}

// ------------------------------------------------------------------------------------------------
// Task 02: image scale down/up.
// ------------------------------------------------------------------------------------------------
void lab01::Task02(const string& srcFile)
{
    string winName = "Task 02: Image pyramid";

    Mat srcImage;
    Mat dstImage;

    // Load the source image
    srcImage = imread(srcFile, 1);

    Show(srcImage, winName, srcImage.cols, srcImage.rows);

    dstImage = lab01::ImagePyramid(&pyrDown, srcImage, 3, 0.5f, winName, srcImage.cols, srcImage.rows);
    dstImage = lab01::ImagePyramid(&pyrUp, dstImage, 3, 2.0f, winName, srcImage.cols, srcImage.rows);
}

// ------------------------------------------------------------------------------------------------
// Task 03: draw image.
// ------------------------------------------------------------------------------------------------
void lab01::Task03()
{
    string winName = "Task 03: Draw image";

    Mat image(200, 200, CV_8UC3);

    Show(image, winName, image.cols, image.rows);

    // draw a circle at (100, 100) with a radius of 20
    Point center(100, 100);
    int radius = 20;
    CvScalar circleColor = { 255, 0, 0, 0 };
    int thickness = -1;
    circle(image, center, radius, circleColor, thickness);

    Show(image, winName, image.cols, image.rows);
    
    // draw a rectangle between (30, 60) and (100, 100)
    for (int row = 60; row < 100; row++)
        for (int col = 30; col < 100; col++)
        {
            image.at<Vec3b>(row, col)[1] = 255;

            // uncomment to draw an opaque green rectangle
            //image.at<Vec3b>(row, col)[0] = 0;
            //image.at<Vec3b>(row, col)[2] = 0;
        }

    Show(image, winName, image.cols, image.rows);
}
