#include <iostream>
#include <string>

#include "lab02.h"

#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace common;

// full path to the root folder of the executable
string root = "";

void main(int argc, char** argv)
{
    cout << "Lab02 - Dan Petre" << endl << endl;

    root = FileFolder(argv[0]);

    if (argc < 3)
    {
        cout << "-t02   <image_path>" << " ... " << "image required by task 02." << endl;
        cout << "-t03.1 <image_path>" << " ... " << "first image required by task 03." << endl;
        cout << "-t03.2 <image_path>" << " ... " << "second image required by task 03." << endl;
        cout << endl;
    }

    string t02Image = "";
    string t03Image1 = "";
    string t03Image2 = "";

    // read the cmd line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-t02") == 0) { t02Image = argv[++i]; }
        if (strcmp(argv[i], "-t03.1") == 0) { t03Image1 = argv[++i]; }
        if (strcmp(argv[i], "-t03.2") == 0) { t03Image2 = argv[++i]; }
    }

    // make sure the files exist
    if (!Exists(t02Image)) { cout << "Please provide a valid image for task 02." << endl; return; }
    if (!Exists(t03Image1)) { cout << "Please provide a valid first image for task 03." << endl; return; }
    if (!Exists(t03Image2)) { cout << "Please provide a valid second image for task 03." << endl; return; }

    lab02::Task02(t02Image);
    lab02::Task03(t03Image1, t03Image2);
}

// ------------------------------------------------------------------------------------------------
// Task 02: Erode/Dilate/Open/Close.
// ------------------------------------------------------------------------------------------------
void lab02::Task02(const string& t02ImageFile)
{
    string winName = "Erode/Dilate/Open/Close";

    Mat srcImage = ReadImage(t02ImageFile, "Original image");
    destroyAllWindows();

    int size = 5;
    ErodeDilateTestAll(srcImage, size);
    destroyAllWindows();

    Open(srcImage, MORPH_ELLIPSE, size);
    destroyAllWindows();

    Close(srcImage, MORPH_ELLIPSE, size);
    destroyAllWindows();

    DoGradientAll(srcImage, MORPH_ELLIPSE, size);
    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Task 03: Coffee cup.
// ------------------------------------------------------------------------------------------------
void lab02::Task03(const string& t03Image1File, const string& t03Image2File)
{
    string winName = "Task03 - Coffee cup";
    string folder = root + "\\..\\..\\images\\lab02\\task03\\";
    string fileName = "";

    Mat srcImage1;
    Mat srcImage2;

    // Load the source images
    srcImage1 = imread(t03Image1File, 1);
    srcImage2 = imread(t03Image2File, 1);

    Show(srcImage1, winName, srcImage1.cols, srcImage1.rows);
    Show(srcImage2, winName, srcImage2.cols, srcImage2.rows);

    // convert input images to CV_8UC1 - 8bit or unsigned char single channel
    Mat grayImage1 = ToGray(srcImage1, winName);
    fileName = folder + "01.Coffee cup - before.jpg";
    Save(grayImage1, fileName);

    Mat grayImage2 = ToGray(srcImage2, winName);
    fileName = folder + "02.Coffee cup - after.jpg";
    Save(grayImage2, fileName);

    // double check the image type
    bool done = CV_8UC1 == grayImage1.type();
    if (done)
        cout << "The images were converted to 8bit grayscale." << endl;

    // calculate the absolute difference and use median blur to attenuate reflections
    Mat t03aImage = Task03a(grayImage1, grayImage2, folder, winName);

    // filter noise & generate cup mask
    Mat t03bImage = Task03b(t03aImage, folder, winName);

    // flood fill test
    Task03c(t03bImage, folder, winName);

    // Task03.e
    Task03e(srcImage1, srcImage2, t03bImage, folder, winName);

    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// Convert to gray.
// ------------------------------------------------------------------------------------------------
Mat lab02::ToGray(const Mat& image, const string& winName, const bool& show)
{
    Mat grayImage;

    cvtColor(image, grayImage, CV_RGB2GRAY);

    // show the gray image
    if (show)
        Show(grayImage, winName, image.cols, image.rows);

    return grayImage;
}

// ------------------------------------------------------------------------------------------------
// Get the negative.
// ------------------------------------------------------------------------------------------------
Mat lab02::ToNeg(const Mat& image, const string& winName, const bool& show)
{
    Mat negImage;

    Size size = { image.cols, image.rows };

    Mat whiteImage = Mat::ones(size, image.type()) * 255;
    subtract(whiteImage, image, negImage);

    // show the negative image
    if (show)
        Show(negImage, winName, size);

    return negImage;
}

// ------------------------------------------------------------------------------------------------
// Convert to a binary image.
// ------------------------------------------------------------------------------------------------
Mat lab02::ToBW(const Mat& image, const int& thresholdType, const string& winName, const bool& show)
{
    Mat bwImage;

    Size size = { image.cols, image.rows };

    // binary threshold
    threshold(image, bwImage, 128, 255, thresholdType);

    // show the binary image
    if (show)
        Show(bwImage, winName, size);

    return bwImage;
}

// ------------------------------------------------------------------------------------------------
// Perform the Erode/Dilate test for the given Erode/Dilate kernel size and type.
// bwOperation is performed on the bw/black-white image.
// bwNegOperation is performed on the negative of the bw/black-white image.
// ------------------------------------------------------------------------------------------------
void lab02::ErodeDilateTest(const Morphology::Type& bwOperation, const Mat& bwImage, const Morphology::Type& bwNegOperation, const Mat& bwNegImage, const int& structuringElement, const int& size)
{
    string winName = "Task 02 - Erode/Dilate test";
    string folder = root + "\\..\\..\\images\\lab02\\task02\\";
    string fileName = "";

    Size ksize = Size(size, size);
    Point anchor = Point((size - 1) / 2, (size - 1) / 2);
    Mat kernel = getStructuringElement(structuringElement, ksize, anchor);

    Mat imageA;
    Mat imageB;

    switch (bwOperation)
    {
    case Morphology::Type::DILATE: dilate(bwImage, imageA, kernel); break;
    case Morphology::Type::ERODE: erode(bwImage, imageA, kernel); break;
    }

    switch (bwNegOperation)
    {
    case Morphology::Type::DILATE: dilate(bwNegImage, imageB, kernel); break;
    case Morphology::Type::ERODE: erode(bwNegImage, imageB, kernel); break;
    }

    fileName = folder + ErodeDilateWinName(bwOperation, structuringElement, false) + ".jpg";
    Show(imageA, winName, imageA.cols, imageA.rows);
    Save(imageA, fileName);

    fileName = folder + ErodeDilateWinName(bwNegOperation, structuringElement, true) + ".jpg";
    Mat imageBNeg = ToNeg(imageB, winName);
    Show(imageB, winName, imageB.cols, imageB.rows);
    Save(imageBNeg, fileName);

    fileName = folder + ErodeDilateWinName(bwOperation, bwNegOperation, structuringElement) + ".jpg";
    Mat diff = imageA != imageBNeg;
    Show(diff, winName, diff.cols, diff.rows);
    Save(diff, fileName);
}

// ------------------------------------------------------------------------------------------------
// Generate all combinations of erode, dilate operations with rect, cross & ellipse structuring elements.
// ------------------------------------------------------------------------------------------------
void lab02::ErodeDilateTestAll(const Mat& image, const int& size)
{
    Mat grayImage = ToGray(image, "Image converted to gray", false);

    // bw = black-white image
    Mat bwImage = ToBW(grayImage, CV_THRESH_BINARY, "Binary threshold image", false);

    // bwNeg = negative of black-white image
    Mat bwNegImage = ToBW(grayImage, CV_THRESH_BINARY_INV, "Binary threshold image inverse/negative", false);

    // The ErodeDilateTest
    // Generate all combinations of erode, dilate operations on black&white image and black&white negative image
    // with three structuring elements: rect, cross, ellipse.
    // The output of each operations is shown followed by the difference between the two.
    for (int bwOperation = 0; bwOperation < 2; bwOperation++)
    {
        for (int bwNegOperation = 0; bwNegOperation < 2; bwNegOperation++)
        {
            for (int structuringElement = 0; structuringElement < 3; structuringElement++)
            {
                ErodeDilateTest((Morphology::Type)bwOperation, bwImage, (Morphology::Type)bwNegOperation, bwNegImage, structuringElement, size);
                destroyAllWindows();
            }
        }
    }

}
// ------------------------------------------------------------------------------------------------
// Generate the window name for the Erode/Dilate test.
// ------------------------------------------------------------------------------------------------
string lab02::ErodeDilateWinName(const Morphology::Type& bwOperation, const Morphology::Type& bwNegOperation, const int& structuringElement)
{
    stringstream winName;

    winName << "Diference of ";
    winName << Morphology::ToString(bwOperation);
    winName << " bw image and ";
    winName << Morphology::ToString(bwNegOperation);
    winName << " neg bw image with ";

    switch (structuringElement)
    {
    case MORPH_RECT: winName << "a rectangular"; break;
    case MORPH_CROSS: winName << "a cross"; break;
    case MORPH_ELLIPSE: winName << "an ellipse"; break;
    }
    winName << " kernel";

    return winName.str();
}

// ------------------------------------------------------------------------------------------------
// Generate the window name for the Erode/Dilate test.
// ------------------------------------------------------------------------------------------------
string lab02::ErodeDilateWinName(const Morphology::Type& operation, const int& structuringElement, const bool& isNegImage)
{
    stringstream winName;

    winName << Morphology::ToString(operation);
    winName << (isNegImage ? " neg bw image" : " bw image");

    winName << " with ";
    switch (structuringElement)
    {
    case MORPH_RECT: winName << "a rectangular"; break;
    case MORPH_CROSS: winName << "a cross"; break;
    case MORPH_ELLIPSE: winName << "an ellipse"; break;
    }
    winName << " kernel";

    return winName.str();
}

// ------------------------------------------------------------------------------------------------
// Open an image: erode followed by dilate.
// ------------------------------------------------------------------------------------------------
void lab02::Open(const Mat& image, const int& structuringElement, const int& size)
{
    string winName = "Task 02 - Opening";
    string folder = root + "\\..\\..\\images\\lab02\\task02\\";
    string fileName = "";

    Mat grayImage = ToGray(image, "Image converted to gray", false);

    // bw = black-white image
    fileName = folder + "Opening - binary threshold image" + ".jpg";
    Mat bwImage = ToBW(grayImage, CV_THRESH_BINARY, winName);
    Save(bwImage, fileName);

    Size ksize = Size(size, size);
    Point anchor = Point((size - 1) / 2, (size - 1) / 2);
    Mat kernel = getStructuringElement(structuringElement, ksize, anchor);

    Mat erodedImage;
    Mat openedImage;

    erode(bwImage, erodedImage, kernel);
    dilate(erodedImage, openedImage, kernel);

    fileName = folder + "Opening 1st stage - erode" + ".jpg";
    Show(erodedImage, winName, erodedImage.cols, erodedImage.rows);
    Save(erodedImage, fileName);

    fileName = folder + "Opening 2nd stage - dilate" + ".jpg";
    Show(openedImage, winName, openedImage.cols, openedImage.rows);
    Save(openedImage, fileName);
}

// ------------------------------------------------------------------------------------------------
// Close an image: dilate followed by erode.
// ------------------------------------------------------------------------------------------------
void lab02::Close(const Mat& image, const int& structuringElement, const int& size)
{
    string winName = "Task 02 - Closing";
    string folder = root + "\\..\\..\\images\\lab02\\task02\\";
    string fileName = "";

    Mat grayImage = ToGray(image, "Image converted to gray", false);

    // bw = black-white image
    fileName = folder + "Closing - binary threshold image" + ".jpg";
    Mat bwImage = ToBW(grayImage, CV_THRESH_BINARY, winName);
    Save(bwImage, fileName);

    Size ksize = Size(size, size);
    Point anchor = Point((size - 1) / 2, (size - 1) / 2);
    Mat kernel = getStructuringElement(structuringElement, ksize, anchor);

    Mat dilatedImage;
    Mat openedImage;

    dilate(bwImage, dilatedImage, kernel);
    erode(dilatedImage, openedImage, kernel);

    fileName = folder + "Closing 1st stage - dilate" + ".jpg";
    Show(dilatedImage, winName, dilatedImage.cols, dilatedImage.rows);
    Save(dilatedImage, fileName);

    fileName = folder + "Closing 2nd stage - erode" + ".jpg";
    Show(openedImage, winName, openedImage.cols, openedImage.rows);
    Save(openedImage, fileName);
}

// ------------------------------------------------------------------------------------------------
// Morphological gradient:
// 1. dilated image - eroded image
// 2. original image - eroded image
// 3. dilated image - original image
// ------------------------------------------------------------------------------------------------
void lab02::DoGradient(const Mat& image, const Gradient::Type& type, const bool& doNegative, const int& structuringElement, const int& size)
{
    string winName = "Task 02 - Morphological gradient";
    string folder = root + "\\..\\..\\images\\lab02\\task02\\";
    string fileName = "";

    Mat grayImage = ToGray(image, "Image converted to gray", false);

    Mat bwImage;

    if (doNegative)
    {
        fileName = folder + "Gradient - binary threshold image negative" + ".jpg";
        bwImage = ToBW(grayImage, CV_THRESH_BINARY_INV, winName, false);
    }
    else
    {
        fileName = folder + "Gradient - binary threshold image" + ".jpg";
        bwImage = ToBW(grayImage, CV_THRESH_BINARY, winName);
    }
    Save(bwImage, fileName);

    Size ksize = Size(size, size);
    Point anchor = Point((size - 1) / 2, (size - 1) / 2);
    Mat kernel = getStructuringElement(structuringElement, ksize, anchor);

    Mat imageA;
    Mat imageB;
    Mat gradient;

    switch (type)
    {
    case Gradient::Type::DILATED_ERODED:
    {
        dilate(bwImage, imageA, kernel);
        erode(imageA, imageB, kernel);

        break;
    }
    case Gradient::Type::ORIGINAL_ERODED:
    {
        imageA = bwImage.clone();
        erode(imageA, imageB, kernel);

        break;
    }
    case Gradient::Type::DILATED_ORIGINAL:
    {
        dilate(bwImage, imageA, kernel);
        imageB = bwImage.clone();

        break;
    }
    }

    gradient = imageA - imageB;

    string imageType = doNegative ? " negative image" : " true image";

    fileName = folder + "Gradient " + Gradient::ToString(type) + imageType + " - 1st stage " + ".jpg";
    Show(imageA, winName, imageA.cols, imageA.rows);
    Save(imageA, fileName);

    fileName = folder + "Gradient " + Gradient::ToString(type) + imageType + " - 2nd stage " + ".jpg";
    Show(imageB, winName, imageB.cols, imageB.rows);
    Save(imageB, fileName);

    fileName = folder + "Gradient " + Gradient::ToString(type) + imageType + " - result " + ".jpg";
    Show(gradient, winName, gradient.cols, gradient.rows);
    Save(gradient, fileName);
}

// ------------------------------------------------------------------------------------------------
// Perform all morphological gradient combinations.
// ------------------------------------------------------------------------------------------------
void lab02::DoGradientAll(const cv::Mat& image, const int& structuringElement, const int& size)
{
    bool doNegative = false;
    DoGradient(image, Gradient::Type::DILATED_ERODED, doNegative, MORPH_ELLIPSE, size);
    destroyAllWindows();

    DoGradient(image, Gradient::Type::ORIGINAL_ERODED, doNegative, MORPH_ELLIPSE, size);
    destroyAllWindows();

    DoGradient(image, Gradient::Type::DILATED_ORIGINAL, doNegative, MORPH_ELLIPSE, size);
    destroyAllWindows();

    doNegative = true;
    DoGradient(image, Gradient::Type::DILATED_ERODED, doNegative, MORPH_ELLIPSE, size);
    destroyAllWindows();

    DoGradient(image, Gradient::Type::ORIGINAL_ERODED, doNegative, MORPH_ELLIPSE, size);
    destroyAllWindows();

    DoGradient(image, Gradient::Type::DILATED_ORIGINAL, doNegative, MORPH_ELLIPSE, size);
    destroyAllWindows();
}

// ------------------------------------------------------------------------------------------------
// calculate the absolute difference and use median blur to attenuate reflections
// ------------------------------------------------------------------------------------------------
Mat lab02::Task03a(const Mat& grayImage1, const Mat& grayImage2, const string& folder, const string& winName)
{
    // Task03.a
    Mat absDiffImage;
    absdiff(grayImage1, grayImage2, absDiffImage);
    string fileName = folder + "03.Coffee cup  - abs difference.jpg";
    Show(absDiffImage, winName, absDiffImage.cols, absDiffImage.rows);
    Save(absDiffImage, fileName);

    // perform a blur to remove some of the reflections
    Mat blurImage;
    int ksize = 7;
    medianBlur(absDiffImage, blurImage, ksize);
    Show(blurImage, winName, blurImage.cols, blurImage.rows);
    fileName = folder + "03.a.Coffee cup - blur.jpg";
    Save(blurImage, fileName);

    return blurImage;
}

// ------------------------------------------------------------------------------------------------
// filter noise & generate cup mask
// ------------------------------------------------------------------------------------------------
Mat lab02::Task03b(const Mat& blurImage, const string& folder, const string& winName)
{
    // Task03.b
    Mat thrImage;
    threshold(blurImage, thrImage, 65, 255, THRESH_BINARY);
    string fileName = folder + "04.Coffee cup - threshold.jpg";
    Show(thrImage, winName, thrImage.cols, thrImage.rows);
    Save(thrImage, fileName);

    // peform erode/dilate to eliminate the rest of the artifacts from reflections
    int size = 7;
    Size morfSize = Size(size, size);
    Point anchor = Point((size - 1) / 2, (size - 1) / 2);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, morfSize, anchor);

    Mat erdImage;
    erode(thrImage, erdImage, kernel, anchor);
    Show(erdImage, winName, erdImage.cols, erdImage.rows);
    fileName = folder + "04.a.Coffee cup - erode.jpg";
    Save(erdImage, fileName);

    Mat dilImage;
    dilate(erdImage, dilImage, kernel, anchor);
    Show(dilImage, winName, dilImage.cols, dilImage.rows);
    fileName = folder + "04.b.Coffee cup - dilate.jpg";
    Save(dilImage, fileName);

    return dilImage;
}

// ------------------------------------------------------------------------------------------------
// flood fill test
// ------------------------------------------------------------------------------------------------
void lab02::Task03c(const Mat& t03bImage, const string& folder, const string& winName)
{
    Mat fillImage = t03bImage.clone();
    int row = 0;
    int col = 0;
    bool found = false;
    for (row = 0; row < fillImage.rows; row++)
    {
        for (col = 0; col < fillImage.cols; col++)
        {
            if (fillImage.ptr<uchar>(row)[col] == 255)
            {
                found = true;
                break;
            }
        }

        if (found)
            break;
    }
    floodFill(fillImage, Point(col, row), 100);
    Show(fillImage, winName, fillImage.cols, fillImage.rows);
    string fileName = folder + "05.Coffee cup - flood fill.jpg";
    Save(fillImage, fileName);
}

// ------------------------------------------------------------------------------------------------
// perform the image composition using bitwise operations.
// ------------------------------------------------------------------------------------------------
void lab02::Task03e(const Mat& srcImage1, const Mat& srcImage2, const Mat& t03bImage, const string& folder, const string& winName)
{
    Mat image = ReadImage("C:\\OpenCV\\sources\\samples\\cpp\\board.jpg", winName);
    Rect region(0, 0, srcImage1.cols, srcImage1.rows);
    Mat croppedImage = image(region);

    Mat mask;
    cvtColor(t03bImage, mask, CV_GRAY2RGB);
    Show(mask, winName, mask.cols, mask.rows);

    Mat maskComplement;
    cvtColor(255 - t03bImage, maskComplement, CV_GRAY2RGB);
    Show(maskComplement, winName, maskComplement.cols, maskComplement.rows);

    Mat stage1;
    bitwise_and(srcImage2, mask, stage1);
    Show(stage1, winName, stage1.cols, stage1.rows);
    string fileName = folder + "06.a.Coffee cup - just the cup.jpg";
    Save(stage1, fileName);

    Mat stage2;
    bitwise_and(croppedImage, maskComplement, stage2);
    Show(stage2, winName, stage2.cols, stage2.rows);
    fileName = folder + "06.b.Coffee cup - image without the cup.jpg";
    Save(stage2, fileName);

    Mat output;
    bitwise_or(stage1, stage2, output);
    Show(output, winName, output.cols, output.rows);
    fileName = folder + "06.Coffee cup - composite.jpg";
    Save(output, fileName);
}
