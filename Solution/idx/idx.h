#ifndef IDX_H
#define IDX_H

#include <string>
#include <cstdint>

class idx
{
public:
    // Initialization constructor.
    idx(const std::string& filePath);

    // Cleanup.
    ~idx();

private:
    // remove default/copy constructor and assignment operator
    idx() = delete;
    idx(const idx& obj) = delete;
    idx(const idx *const obj) = delete;
    idx(idx *const obj) = delete;
    idx& operator=(const idx& obj) = delete;
    idx& operator=(const idx *const obj) = delete;
    idx& operator=(idx *const obj) = delete;

protected:
    // data size in bytes
    uint32_t size;
    // idx data
    uint8_t* data;

private:
    // Read the file contents.
    uint8_t* read(const std::string& filePath);

    // interpret the file contents.
    virtual void interpret(const uint8_t *const data) = 0;
};

#endif