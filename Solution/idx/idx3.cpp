#include "idx3.h"

// ------------------------------------------------------------------------------------------------
// Initialization constructor.
// ------------------------------------------------------------------------------------------------
idx3::idx3(const std::string& filePath) : idx(filePath)
{
    interpret(data);
}

// ------------------------------------------------------------------------------------------------
// 
// ------------------------------------------------------------------------------------------------
idx3::~idx3()
{
    delete[] pixels;
    pixels = NULL;
}

typedef union _Converter
{
    int8_t bytes[4];
    int value;
} Converter;

// ------------------------------------------------------------------------------------------------
// interpret the file contents.
// ------------------------------------------------------------------------------------------------
void idx3::interpret(const uint8_t *const data)
{
    Converter c;

    int32_t offset = 0;

    c.bytes[0] = data[offset + 3];
    c.bytes[1] = data[offset + 2];
    c.bytes[2] = data[offset + 1];
    c.bytes[3] = data[offset + 0];
    magicNumber = c.value;

    offset = 4;
    c.bytes[0] = data[offset + 3];
    c.bytes[1] = data[offset + 2];
    c.bytes[2] = data[offset + 1];
    c.bytes[3] = data[offset + 0];
    nrImages = c.value;

    offset = 8;
    c.bytes[0] = data[offset + 3];
    c.bytes[1] = data[offset + 2];
    c.bytes[2] = data[offset + 1];
    c.bytes[3] = data[offset + 0];
    rows = c.value;

    offset = 12;
    c.bytes[0] = data[offset + 3];
    c.bytes[1] = data[offset + 2];
    c.bytes[2] = data[offset + 1];
    c.bytes[3] = data[offset + 0];
    cols = c.value;

    offset = 16;
    int32_t nrPixels = nrImages * rows * cols;
    pixels = new float[nrPixels];

    // convert uint8_t to float and scale to a value between 0.0f and 1.0f
    for (int i = 0; i < nrPixels; i++)
        pixels[i] = (float)data[offset + i] / 256.0f;
}

// ------------------------------------------------------------------------------------------------
// Return a pointer to the image at the specified index.
// ------------------------------------------------------------------------------------------------
const float *const idx3::operator[](const int32_t& index) const
{
    return &pixels[index * rows * cols];
}
