
#include "common.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

// output on key press switch
bool common::output = true;

// ------------------------------------------------------------------------------------------------
// Get file folder.
// Works fine in VS debug.
// Will fail if the path of the exe from argv** is not be complete.
// ------------------------------------------------------------------------------------------------
string common::FileFolder(const string& filePath)
{
    string output = filePath;

    size_t index = output.find_last_of("\\");

    if (string::npos == index)
        return "";

    return move(output.erase(index));
};

// ------------------------------------------------------------------------------------------------
// Get file name with ext.
// ------------------------------------------------------------------------------------------------
string common::FileNameWithExt(const string& filePath)
{
    string output = filePath;

    size_t index = output.find_last_of("\\");

    return move(output.erase(0, index));
};

// ------------------------------------------------------------------------------------------------
// Get file name without ext.
// ------------------------------------------------------------------------------------------------
string common::FileName(const string& filePath)
{
    string output = filePath;

    size_t begin = output.find_last_of("\\");
    begin = (string::npos == begin) ? 0 : begin + 1;

    size_t end = output.find_last_of(".");
    end = (string::npos == end) ? output.length() : end;

    return move(output.substr(begin, end - begin));
};

// ------------------------------------------------------------------------------------------------
// Check if the file exits.
// ------------------------------------------------------------------------------------------------
bool common::Exists(const string& filePath)
{
    ifstream fileStream(filePath.c_str());

    return fileStream.is_open();
}

// ------------------------------------------------------------------------------------------------
// Concatenate a directory path with a fileName.
// ------------------------------------------------------------------------------------------------
string common::Concatenate(const string& directory, const string& fileName)
{
    stringstream output;

    output << directory;

    if ('\\' != directory.back())
        output << "\\";

    output << fileName;

    return output.str();
}

// ------------------------------------------------------------------------------------------------
// Display the image.
// ------------------------------------------------------------------------------------------------
int common::Show
(
const InputArray& image,
const string& winName,
const int& cols,
const int& rows,
const int& flags,
const int& posx,
const int& posy
)
{
    if (!output)
        return -1;

    namedWindow(winName, flags);
    imshow(winName, image);
    resizeWindow(winName, cols, rows);

    if (0 != posx && 0 != posy)
        moveWindow(winName, posx, posy);

    return waitKey();
}

// ------------------------------------------------------------------------------------------------
// Display the image.
// ------------------------------------------------------------------------------------------------
int common::Show
(
const InputArray& image,
const string& winName,
const Size& size,
const int& flags,
const cv::Point& pos
)
{
    return Show(image, winName, size.width, size.height, flags, pos.x, pos.y);
}

// ------------------------------------------------------------------------------------------------
// Save image to disk.
// ------------------------------------------------------------------------------------------------
bool common::Save(const cv::InputArray& image, const string& dstFile)
{
    if (!imwrite(dstFile, image))
    {
        cout << "Could not open the output file: " << dstFile << endl;

        return false;
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
// Read the image from the disk.
// ------------------------------------------------------------------------------------------------
Mat common::ReadImage(const string& imageFile, const string& winName, const bool& show)
{
    Mat image = imread(imageFile);

    // show the input image
    if (show)
        Show(image, winName, image.cols, image.rows);

    return image;
}
