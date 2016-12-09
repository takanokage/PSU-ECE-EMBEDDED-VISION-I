#ifndef CLOCK_H
#define CLOCK_H

#include <cstdint>

class clock
{
private:
    // Start counter.
    static int64_t startCounter;
    // Stop counter.
    static int64_t stopCounter;

public:
    // Set the start marker.
    static void start();
    // Set the stop marker.
    static void stop();
    // Calculate the elapsed time in ms between the start and stop markers.
    static double delta();

private:
    // no constructors, no destructor
    clock() = delete;
    ~clock() = delete;
};

#endif
