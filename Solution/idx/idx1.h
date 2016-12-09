
#ifndef IDX_1_H
#define IDX_1_H

#include "idx.h"
class idx1 : public idx
{
public:
    // Initialization constructor.
    idx1(const std::string& filePath);

    // Cleanup.
    ~idx1();

private:
    // remove default/copy constructor and assignment operator
    idx1() = delete;
    idx1(const idx1& gen) = delete;
    idx1(const idx1 *const gen) = delete;
    idx1(idx1 *const gen) = delete;
    idx1& operator=(const idx1& gen) = delete;
    idx1& operator=(const idx1 *const gen) = delete;
    idx1& operator=(idx1 *const gen) = delete;

private:
    int32_t magicNumber;
    int32_t nrItems;
    float* labels;

private:
    // interpret the file contents.
    void interpret(const uint8_t *const data);

public:
    // Get the total number of images in this database.
    int32_t Count() const { return nrItems; };

    // Return the value at the specified index.
    float operator[](const int32_t& index) const;

    // Return a pointer to all the labels.
    const float *const ptr() const { return labels; }
};

#endif