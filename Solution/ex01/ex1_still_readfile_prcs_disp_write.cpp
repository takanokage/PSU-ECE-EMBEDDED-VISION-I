
// ex1: reads a still image file, does some simple processing
//      displaying input, intermediate and output images
//      then writes final image to a file
//
//      reads 1 command line parameter, source file name

#include <iostream> // for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "common.h"

using namespace std;
using namespace cv;
using namespace common;

int main(int argc, char** argv)
{
    cout << "ex1" << endl;

    // get the exe path
    string root = FileFolder(argv[0]);

    Mat src; Mat prcs;
    string win_src = "Image_orig";
    string win_prcs = "Image_prcs";

    /// Load the source image
    const string src_file = argv[1]; // the source file name
    src = imread(src_file, 1);

    namedWindow(win_src, CV_WINDOW_AUTOSIZE);
    imshow(win_src, src);

    prcs = src.clone();
    GaussianBlur(src, prcs, Size(15, 15), 0, 0);

    namedWindow(win_prcs, CV_WINDOW_AUTOSIZE);
    imshow(win_prcs, prcs);

    // form output file name from input name, use file type jpg
    const string prcs_file = Concatenate(root, FileName(src_file) + "2" + ".jpg");   // Form the new name with container

    // now write to file
    if (!imwrite(prcs_file, prcs))
    {
        cout << "Could not open the output file: " << prcs_file << endl;
        return -1;
    }

    waitKey();  // don't exit, wait for user to view images and then type key

    destroyAllWindows();

    return 0;
}

