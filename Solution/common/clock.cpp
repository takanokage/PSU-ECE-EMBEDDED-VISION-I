#include "clock.h"

#include <Windows.h>

// ------------------------------------------------------------------------------------------------
// Start counter.
// ------------------------------------------------------------------------------------------------
int64_t clock::startCounter = 0;

// ------------------------------------------------------------------------------------------------
// Stop counter.
// ------------------------------------------------------------------------------------------------
int64_t clock::stopCounter = 0;

// ------------------------------------------------------------------------------------------------
// Set the start marker.
// ------------------------------------------------------------------------------------------------
void clock::start()
{
    LARGE_INTEGER counter;

    QueryPerformanceCounter(&counter);

    startCounter = counter.QuadPart;
}

// ------------------------------------------------------------------------------------------------
// Set the stop marker.
// ------------------------------------------------------------------------------------------------
void clock::stop()
{
    LARGE_INTEGER counter;

    QueryPerformanceCounter(&counter);

    stopCounter = counter.QuadPart;
}

// ------------------------------------------------------------------------------------------------
// Calculate the elapsed time in ms between the start and stop markers.
// ------------------------------------------------------------------------------------------------
double clock::delta()
{
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    double ms = (double)(stopCounter - startCounter) / frequency.QuadPart * 1000.0;

    return ms;
}
