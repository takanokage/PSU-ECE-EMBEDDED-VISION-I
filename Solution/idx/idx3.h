
#ifndef IDX_3_H
#define IDX_3_H

#include "idx.h"
class idx3 : public idx
{
public:
    // Initialization constructor.
    idx3(const std::string& filePath);

    // Cleanup.
    ~idx3();

private:
    // remove default/copy constructor and assignment operator
    idx3() = delete;
    idx3(const idx3& gen) = delete;
    idx3(const idx3 *const gen) = delete;
    idx3(idx3 *const gen) = delete;
    idx3& operator=(const idx3& gen) = delete;
    idx3& operator=(const idx3 *const gen) = delete;
    idx3& operator=(idx3 *const gen) = delete;

private:
    int32_t magicNumber;
    int32_t nrImages;
    int32_t rows;
    int32_t cols;
    float* pixels;

private:
    // interpret the file contents.
    void interpret(const uint8_t *const data);

public:
    // Get the total number of images in this database.
    int32_t Count() const { return nrImages; };
    // Get then number of rows.
    int32_t Rows() const { return rows; };
    // Get then number of columns.
    int32_t Cols() const { return cols; };

    // Return a pointer to the image at the specified index.
    const float *const operator[](const int32_t& index) const;

    // Return a pointer to all the images.
    const float *const ptr() const { return pixels; }
};

#endif