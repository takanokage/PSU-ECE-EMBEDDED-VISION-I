 /*------------------------------------------------------------------------------------------*\
 This file contains material supporting chapter 4 of the cookbook:
 Computer Vision Programming using the OpenCV Library.
 by Robert Laganiere, Packt Publishing, 2011.
 Copyright (C) 2010-2011 Robert Laganiere, www.laganiere.name
 \*------------------------------------------------------------------------------------------*/


//////////////////////////// Object finder with Histograms //////////////////////////////////


#include <iostream> // for standard I/O
#include <string>   // for strings
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

#include "histogram.h"
#include "objectFinder.h"
#include "colorhistogram.h"

int main()
{
    // Read input image ------------------------------
    cv::Mat image = cv::imread("waves.jpg",cv::IMREAD_GRAYSCALE); // 0 means gray scale
    if (!image.data)
        return 0;
    
    // define image ROI
    cv::Mat imageROI;
    imageROI = image(cv::Rect(360,55,40,50)); // Cloud region
    
    // Display reference patch
    cv::namedWindow("Reference");
    cv::imshow("Reference ROI",imageROI);
    cv::waitKey();

    // Find histogram of reference
    Histogram1D h;
    cv::MatND hist = h.getHistogram(imageROI);
    cv::namedWindow("Reference ROI Hist");
    cv::imshow("Reference ROI Hist",h.getHistogramImage(imageROI));
    cv::waitKey();

    // Create the objectfinder object
    ObjectFinder finder;
    finder.setHistogram(hist);
    finder.setThreshold(-1.0f);
    
    // Get back-projection
    cv::Mat result1;
    result1 = finder.find(image);
    
    // Create negative image and display result
    cv::Mat tmp;
    result1.convertTo(tmp,CV_8U,-1.0,255.0);
    // -1.0 is a scale factor, 255.0 is added to each pixel, to get complementary value
    cv::namedWindow("Monochrome Backprojection result");
    cv::imshow("Monochrome Backprojection result",tmp);
    cv::waitKey();

    // Get binary back-projection
    finder.setThreshold(0.12f);
    result1 = finder.find(image);
    
    // Draw a rectangle around the reference area in original image
    cv::rectangle(image,cv::Rect(360,55,40,50),cv::Scalar(0,0,0));
    
    // Display original image
    cv::namedWindow("Original Image");
    cv::imshow("Original Image",image);
    cv::waitKey();

    // Display thresholded back projected result
    cv::namedWindow("Monochrome Detection Result");
    cv::imshow("Monochrome Detection Result",result1);
    cv::waitKey();

    
    // Second test image, using existing ROI histogram
    // Compute back projection again with new image, same ROI histogram
    cv::Mat image2 = cv::imread("dog.jpg",cv::IMREAD_GRAYSCALE);
    cv::Mat result2;
    result2 = finder.find(image2);
    
    // Display result
    cv::namedWindow("Monochrome Result with 2nd Image");
    cv::imshow("Monochrome Result with 2nd Image",result2);
    cv::waitKey();

    
    
    // Load color image -------------------------------
    ColorHistogram hc;  // instantiate color histogram object
    cv::Mat color = cv::imread("waves.jpg");  // default is color
    color = hc.colorReduce(color,32);
    cv::namedWindow("Original Color Image");
    cv::imshow("Original Color Image",color);
    cv::waitKey();

    imageROI = color(cv::Rect(0,0,165,75)); // blue sky area
    
    // Get 3D color histogram
    cv::MatND shist= hc.getHistogram(imageROI);
    
    finder.setHistogram(shist);
    finder.setThreshold(0.05f);
    
    // Get back-projection of color histogram
    result1 = finder.find(color);
    
    cv::namedWindow("Color Backproject Result");
    cv::imshow("Color Backproject Result",result1);
    cv::waitKey();

    // Second color image
    cv::Mat color2 = cv::imread("dog.jpg");   // default is color
    color2 = hc.colorReduce(color2,32);
    
    // Get back-projection of color histogram for 2nd image
    result2 = finder.find(color2);
    
    cv::namedWindow("Color Backproject Result with 2nd Image");
    cv::imshow("Color Backproject Result with 2nd Image",result2);
    
    cv::waitKey();
    return 0;
}
