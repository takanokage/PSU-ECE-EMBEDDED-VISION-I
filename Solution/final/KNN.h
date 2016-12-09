#ifndef KNN_H
#define KNN_H

#include "idx1.h"
#include "idx3.h"

#include <opencv2/ml.hpp>

#include <vector>

// True + predicted values.
typedef struct _Pair
{
    int32_t trueValue;
    int32_t predicted;
} Pair;

// forward declaration
class KNN_Task;

// KNN classifier
class KNN
{
private:
    friend KNN_Task;

public:
    // Initialization constructor.
    KNN
    (
        const cv::Mat& trnImages,
        const cv::Mat& trnLabels,
        const int32_t& K
    );

    // Cleanup.
    ~KNN();

private:
    // remove default/copy constructor and assignment operator
    KNN() = delete;
    KNN(const KNN& obj) = delete;
    KNN(const KNN *const obj) = delete;
    KNN(KNN *const obj) = delete;
    KNN& operator=(const KNN& obj) = delete;
    KNN& operator=(const KNN *const obj) = delete;
    KNN& operator=(KNN *const obj) = delete;

private:
    cv::Ptr<cv::ml::TrainData> trainingData;
    cv::Mat tstImages;
    cv::Mat tstLabels;

    cv::Ptr<cv::ml::KNearest> classifier;
    cv::ml::SampleTypes sampleType;
    cv::ml::KNearest::Types knnType;
    int32_t truePositives;
    int32_t trueNegatives;
    std::vector<int32_t> trueValue;
    std::vector<int32_t> predicted;

public:
    // Calculate the detection accuracy achieved so far.
    double Accuracy();

    // Compute the predictions for the entire test set.
    void Predict(const cv::Mat& tstImages, const cv::Mat& tstLabels);
    // Compute the predictions for the entire test set. No ground truth is available.
    void Predict(const cv::Mat& tstImages);

    // Return the true-predicted pair corresponding to the specified index.
    Pair operator[](const int32_t& index) const;
};

// KNN single thread
class KNN_Task : public cv::ParallelLoopBody
{
public:
    KNN_Task
    (
        const KNN *const knn,
        const cv::Mat& tstImages,
        int32_t *const predicted
    ) :
        knn(knn),
        tstImages(tstImages),
        predicted(predicted) {}

private:
    const KNN *const knn;
    cv::Mat tstImages;
    int32_t *const predicted;

public:
    // Single thread body.
    virtual void operator()(const cv::Range& range) const;
};

#endif