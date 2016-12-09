#include "KNN.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

// ------------------------------------------------------------------------------------------------
// Initialization constructor.
// ------------------------------------------------------------------------------------------------
KNN::KNN
(
    const Mat& trnImages,
    const Mat& trnLabels,
    const int32_t& K
)
{
    this->tstImages = tstImages;
    this->tstLabels = tstLabels;

    truePositives = 0;
    trueNegatives = 0;

    sampleType = SampleTypes::ROW_SAMPLE;
    knnType = KNearest::Types::BRUTE_FORCE;

    classifier = KNearest::create();
    trainingData = TrainData::create(trnImages, sampleType, trnLabels);

    classifier->setIsClassifier(true);
    classifier->setAlgorithmType(knnType);
    classifier->setDefaultK(K);

    classifier->train(trainingData);
}

// ------------------------------------------------------------------------------------------------
// Destructor.
// ------------------------------------------------------------------------------------------------
KNN::~KNN()
{
}

// ------------------------------------------------------------------------------------------------
// Calculate the detection accuracy achieved so far.
// ------------------------------------------------------------------------------------------------
double KNN::Accuracy()
{
    return (double)truePositives / (truePositives + trueNegatives) * 100.0;
}

// ------------------------------------------------------------------------------------------------
// Compute the predictions for the entire test set.
// ------------------------------------------------------------------------------------------------
void KNN::Predict(const Mat& tstImages, const Mat& tstLabels)
{
    int32_t testSize = tstImages.rows;

    trueValue = vector<int32_t>(testSize, -1);
    predicted = vector<int32_t>(testSize, -2);

    truePositives = 0;
    trueNegatives = 0;

    // the prediction is performed in parallel on the whole test set
    parallel_for_(Range(0, testSize), KNN_Task(this, tstImages, &predicted[0]));

    // evaluation
    for (int32_t row = 0; row < testSize; row++)
    {
        trueValue[row] = (int32_t)tstLabels.at<float>(row, 0);

        if (predicted[row] == trueValue[row])
            truePositives++;
        else
            trueNegatives++;
    }
}

// ------------------------------------------------------------------------------------------------
// Compute the predictions for the entire test set. No ground truth is available.
// ------------------------------------------------------------------------------------------------
void KNN::Predict(const Mat& tstImages)
{
    int32_t testSize = tstImages.rows;

    trueValue = vector<int32_t>(testSize, -1);
    predicted = vector<int32_t>(testSize, -2);

    truePositives = 0;
    trueNegatives = 0;

    // the prediction is performed in parallel on the whole test set
    parallel_for_(Range(0, testSize), KNN_Task(this, tstImages, &predicted[0]));

    // no evaluation
}

// ------------------------------------------------------------------------------------------------
// Return the true-predicted pair corresponding to the specified index.
// ------------------------------------------------------------------------------------------------
Pair KNN::operator[](const int32_t& index) const
{
    return Pair{ trueValue[index], predicted[index] };
}

// ------------------------------------------------------------------------------------------------
// Single thread body.
// ------------------------------------------------------------------------------------------------
void KNN_Task::operator()(const Range& range) const
{
    int32_t K = knn->classifier->getDefaultK();

    for (int i = range.start; i < range.end; i++)
    {
        Mat results;

        predicted[i] = (int)knn->classifier->findNearest(tstImages.row(i), K, results);
    }
}
