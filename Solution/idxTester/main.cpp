
#include <algorithm>
#include <iostream>

using namespace std;

#include "idx1.h"
#include "idx3.h"

void main()
{
    idx1 labels("c:/OpenCV/EBDVI_dpetre/images/mnist/t10k-labels.idx1-ubyte");
    idx3 images("c:/OpenCV/EBDVI_dpetre/images/mnist/t10k-images.idx3-ubyte");
    //idx1 labels("c:/OpenCV/EBDVI_dpetre/images/mnist/train-labels.idx1-ubyte");
    //idx3 images("c:/OpenCV/EBDVI_dpetre/images/mnist/train-images.idx3-ubyte");

    int nrLabels = labels.Count();
    int nrImages = images.Count();
    int count = min(nrLabels, nrImages);

    int rows = images.Rows();
    int cols = images.Cols();

    for (int i = 0; i < count; i++)
    {
        const float* image = images[i];

        cout << "Label: " << labels[i] << endl;

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
                if (image[row * cols + col])
                    cout << "1";
                else
                    cout << "0";

            cout << endl;
        }

        cout << endl;
        cout << endl;
        
        // press any key to continue
        system("pause");
    }
}