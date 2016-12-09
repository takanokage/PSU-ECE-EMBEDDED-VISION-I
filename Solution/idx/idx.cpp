
#include "idx.h"

#include <iostream>
#include <fstream>

using namespace std;

// ------------------------------------------------------------------------------------------------
// Initialization constructor.
// ------------------------------------------------------------------------------------------------
idx::idx(const std::string& filePath) : data(NULL)
{
    data = read(filePath);
}

// ------------------------------------------------------------------------------------------------
// Cleanup.
// ------------------------------------------------------------------------------------------------
idx::~idx()
{
    delete[] data;
}

// ------------------------------------------------------------------------------------------------
// Read the file contents.
// ------------------------------------------------------------------------------------------------
uint8_t* idx::read(const string& filePath)
{
    uint8_t* output = NULL;
    const size_t SIZE = 1024;

    ifstream file(filePath.c_str(), std::ios_base::binary);

    if (!file.good())
        return output;

    file.seekg(0, ios::end);
    size = (uint32_t)file.tellg();

    file.seekg(0, ios::beg);

    output = new uint8_t[size];
    memset(output, 0, size * sizeof(uint8_t));

    int index = 0;
    while (file.good())
    {
        file.read((char*)&output[index], SIZE);

        index += SIZE;
    }

    file.close();

    return output;
}
