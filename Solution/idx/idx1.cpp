#include "idx1.h"

// ------------------------------------------------------------------------------------------------
// Initialization constructor.
// ------------------------------------------------------------------------------------------------
idx1::idx1(const std::string& filePath) : idx(filePath)
{
    interpret(data);
}

// ------------------------------------------------------------------------------------------------
// 
// ------------------------------------------------------------------------------------------------
idx1::~idx1()
{
    delete[] labels;
}

typedef union _Converter
{
    int8_t bytes[4];
    int value;
} Converter;

// ------------------------------------------------------------------------------------------------
// interpret the file contents.
// ------------------------------------------------------------------------------------------------
void idx1::interpret(const uint8_t *const data)
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
    nrItems = c.value;

    offset = 8;
    labels = new float[nrItems];
    for (int i = 0; i < nrItems; i++)
        labels[i] = data[offset + i];
}

// ------------------------------------------------------------------------------------------------
// Return the value at the specified index.
// ------------------------------------------------------------------------------------------------
float idx1::operator[](const int32_t& index) const
{
    return labels[index];
}
