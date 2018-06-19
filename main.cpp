#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

void compress(Mat& input, Mat& output, short blockSize);
void compressBlock(Mat& input, Mat& output, short blockSize, int iblock, int jblock);
void decompress(Mat& input, Mat& output, short blockSize);
void decompressBlock(Mat& input, Mat& output, short blockSize, int iblock, int jblock);

int LEVEL;
void fwt97(double* x,int n);
void iwt97(double* x,int n);
void fwt(cv::Mat& src);
void iwt(cv::Mat& src);

int main( int argc, char** argv )
{
    Mat src;
    Mat_<double> bgr[3];
    Mat_<double> bgr2[3];
    std::string window_name = "Dyplom";
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    if(argc > 1)
    {
        src = imread(argv[1], IMREAD_ANYCOLOR);
        LEVEL = atoi(argv[2]);
    }
    else
    {
        src = imread("test3.png", IMREAD_ANYCOLOR);
        LEVEL = 1;
    }
    if( !src.data )
    {
        std::cerr << "File with given name does not exist\n";
        return -1;
    }

    src.convertTo(src, CV_64FC3);
    src = src/255.0;

    Mat compressed = src.clone();

    switch(src.channels())
    {
    case 1:
        fwt(src);
        imshow(window_name, src);
        waitKey();

        iwt(src);
        imshow(window_name, src);
        waitKey();
        break;
    case 3:
        split(src,bgr);
        fwt(bgr[0]);
        fwt(bgr[1]);
        fwt(bgr[2]);
        merge(bgr, 3, src);

        imshow(window_name, src);
        waitKey();

        split(src,bgr);
        iwt(bgr[0]);
        iwt(bgr[1]);
        iwt(bgr[2]);
        merge(bgr, 3, src);

        imshow(window_name, src);
        waitKey();
        break;
    }

    switch(src.channels())
    {
    case 1:
        compress(src, compressed, 4);

        imshow(window_name, compressed);
        waitKey();

        decompress(compressed, src, 4);

        imshow(window_name, src);
        waitKey(0);
        break;
    case 3:
        split(src, bgr);
        split(src, bgr2);
        compress(bgr[0], bgr2[0], 4);
        compress(bgr[1], bgr2[1], 4);
        compress(bgr[2], bgr2[2], 4);
        merge(bgr2, 3, src);

        imshow(window_name, src);
        waitKey();

        split(src, bgr);
        decompress(bgr[0], bgr2[0], 4);
        decompress(bgr[1], bgr2[1], 4);
        decompress(bgr[2], bgr2[2], 4);
        merge(bgr2, 3, src);

        imshow(window_name, src);
        waitKey();
        break;
    }

    return 0;
}

void fwt(cv::Mat& src)
{
    double* data;
    for(int level = 0; level < LEVEL; ++level)
    {
        data = new double[src.cols>>level];
        for(int i = 0; i < (src.rows>>level); ++i)
        {
            for(int j = 0; j < (src.cols>>level); ++j)
            {
                data[j] = src.at<double>(i,j);
            }
            fwt97(data, (src.cols>>level));
            for(int j = 0; j < (src.cols>>level); ++j)
            {
                src.at<double>(i,j) = data[j];
            }
        }
        delete[] data;
        data = new double[src.rows>>level];
        for(int i = 0; i < (src.cols>>level); ++i)
        {
            for(int j = 0; j < (src.rows>>level); ++j)
            {
                data[j] = src.at<double>(j,i);
            }
            fwt97(data, (src.rows>>level));
            for(int j = 0; j < (src.rows>>level); ++j)
            {
                src.at<double>(j,i) = data[j];
            }
        }
        delete[] data;
    }
}

void iwt(cv::Mat& src)
{
    double* data;
    for(int level = LEVEL - 1; level >= 0; --level)
    {
        data = new double[src.rows>>level];
        for(int i = 0; i < (src.cols>>level); ++i)
        {
            for(int j = 0; j < (src.rows>>level); ++j)
            {
                data[j] = src.at<double>(j,i);
            }
            iwt97(data, (src.rows>>level));
            for(int j = 0; j < (src.rows>>level); ++j)
            {
                src.at<double>(j,i) = data[j];
            }
        }
        delete[] data;
        data = new double[src.cols>>level];
        for(int i = 0; i < (src.rows>>level); ++i)
        {
            for(int j = 0; j < (src.cols>>level); ++j)
            {
                data[j] = src.at<double>(i,j);
            }
            iwt97(data, (src.cols>>level));
            for(int j = 0; j < (src.cols>>level); ++j)
            {
                src.at<double>(i,j) = data[j];
            }
        }
        delete[] data;
    }
}



double *temp = NULL;

void fwt97(double* x,int n) {
    double a;

    a = -1.586134342;
    for(int i = 1; i < n - 2; i += 2)
    {
      x[i] += a * (x[i-1] + x[i+1]);
    }
    x[n-1] += 2 * a * x[n-2];

    a=-0.05298011854;
    for(int i = 2; i < n; i += 2)
    {
      x[i] += a * (x[i-1] + x[i+1]);
    }
    x[0] += 2 * a * x[1];

    a=0.8829110762;
    for(int i = 1; i < n - 2; i += 2)
    {
      x[i] += a * (x[i-1] + x[i+1]);
    }
    x[n-1] += 2 * a * x[n-2];

    a=0.4435068522;
    for(int i = 2; i < n; i += 2)
    {
      x[i] += a * (x[i-1] + x[i+1]);
    }
    x[0] += 2 * a * x[1];

    a=1/1.149604398;
    for(int i = 0; i < n; ++i)
    {
        if(i % 2 == 1)
            x[i] *= a;
        else
            x[i] /= a;
    }

    if(temp == NULL)
        temp = new double[n];

    for (int i = 0; i < n; ++i)
    {
        if (i % 2 == 0)
            temp[i>>1] = x[i];
        else
            temp[(n>>1)+(i>>1)] = x[i];
    }
    for (int i = 0; i < n; ++i)
        x[i] = temp[i];

    delete[] temp;
    temp = NULL;
}

void iwt97(double* x,int n) {
    double a;

    if(temp == NULL)
        temp = new double[n];

    for(int i = 0; i < (n>>1); ++i)
    {
        temp[i<<1] = x[i];
        temp[(i<<1)+1] = x[i+(n>>1)];
    }
    for(int i = 0; i < n; ++i)
        x[i] = temp[i];

    a=1.149604398;
    for(int i = 0; i < n; ++i)
    {
        if(i % 2 == 1)
            x[i] *= a;
        else
            x[i] /= a;
    }

    a = -0.4435068522;
    for(int i = 2; i < n; i += 2)
    {
        x[i] += a * (x[i-1] + x[i+1]);
    }
    x[0] += 2 * a * x[1];

    a=-0.8829110762;
    for(int i = 1; i < n - 2; i += 2)
    {
        x[i] += a * (x[i-1] + x[i+1]);
    }
    x[n-1] += 2 * a * x[n-2];

    a=0.05298011854;
    for(int i = 2; i < n; i += 2)
    {
        x[i] += a * (x[i-1] + x[i+1]);
    }
    x[0] += 2 * a * x[1];

    a=1.586134342;
    for(int i = 1; i < n - 2; i += 2)
    {
        x[i] += a * (x[i-1] + x[i+1]);
    }
    x[n-1] += 2 * a * x[n-2];

    delete[] temp;
    temp = NULL;
}

void compress(Mat& input, Mat& output, short blockSize)
{
    for(int i = 0; i < input.rows; i += blockSize)
    {
        for(int j = 0; j < input.cols; j += blockSize)
        {
            Mat block = input(Rect2d(j, i, blockSize, blockSize));
            compressBlock(block, output, blockSize, i, j);
        }
    }
}



void compressBlock(Mat& input, Mat& output, short blockSize, int iblock, int jblock)
{
    static Mat haar8 = (Mat_<double>(8,8) << 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, -1, -1, -1, -1,
                                    2, 2, -2, -2, 0, 0, 0, 0,
                                    0, 0, 0, 0, 2, 2, -2, -2,
                                    4, -4, 0, 0, 0, 0, 0, 0,
                                    0, 0, 4, -4, 0, 0, 0, 0,
                                    0, 0, 0, 0, 4, -4, 0, 0,
                                    0, 0, 0, 0, 0, 0, 4, -4) / 8.0;
    static Mat haar4 = (Mat_<double>(4,4) << 1, 1, 1, 1,
                                                    1, 1, -1, -1,
                                                    sqrt(2), -sqrt(2), 0, 0,
                                                    0, 0, sqrt(2), -sqrt(2)) * 0.5;

    int rows = output.rows;
    int cols = output.cols;
    Mat result = haar4 * input * haar4.t();
    for(int i = 0; i < blockSize; ++i)
    {
        for(int j = 0; j < blockSize; ++j)
        {
            int k = 0, l = 0;
            while(i >> k > 0)
            {
                k++;
            }
            while(j >> l > 0)
            {
                l++;
            }
            int indexi = 0, indexj = 0;
            if(k == l || (k <= 1 && l <= 1))
            {
                if(k > 0)
                {
                    indexi = ((rows/blockSize)<<(k-1)) + i - (1<<(k-1));
                    indexi += ((iblock/blockSize)<<(k-1));
                }
                else
                    indexi = iblock/blockSize;
                if(l > 0)
                {
                    indexj = ((cols/blockSize)<<(l-1)) + j - (1<<(l-1));
                    indexj += ((jblock/blockSize)<<(l-1));
                }
                else
                    indexj = jblock/blockSize;
            }
            else if(k > l)
            {
                indexi = ((rows/blockSize)<<(k-1)) + i - (1<<(k-1));
                indexi += ((iblock/blockSize)<<(k-1));

                indexj = j + ((jblock/blockSize)<<(k-1));
            }
            else if(k < l)
            {

                indexi = i + ((iblock/blockSize)<<(l-1));

                indexj = ((cols/blockSize)<<(l-1)) + j - (1<<(l-1));
                indexj += ((jblock/blockSize)<<(l-1));
            }
            double* resultrow = result.ptr<double>(i);
            double* outputrow = output.ptr<double>(indexi);
            outputrow[indexj] = resultrow[j];
        }
    }
}

void decompress(Mat& compressed, Mat& decompressed, short blockSize)
{
    for(int i = 0; i < compressed.rows; i += blockSize)
    {
        for(int j = 0; j < compressed.cols; j += blockSize)
        {
            Mat block = decompressed(Rect2d(j, i, blockSize, blockSize));
            decompressBlock(block, compressed, blockSize, i, j);
        }
    }
}

void decompressBlock(Mat& decompressedBlock, Mat& compressed, short blockSize, int iblock, int jblock)
{
    static Mat haar8 = (Mat_<double>(8,8) << 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, -1, -1, -1, -1,
                                    2, 2, -2, -2, 0, 0, 0, 0,
                                    0, 0, 0, 0, 2, 2, -2, -2,
                                    4, -4, 0, 0, 0, 0, 0, 0,
                                    0, 0, 4, -4, 0, 0, 0, 0,
                                    0, 0, 0, 0, 4, -4, 0, 0,
                                    0, 0, 0, 0, 0, 0, 4, -4) / 8.0;
    static Mat haar4 = (Mat_<double>(4,4) << 1, 1, 1, 1,
                                                    1, 1, -1, -1,
                                                    sqrt(2), -sqrt(2), 0, 0,
                                                    0, 0, sqrt(2), -sqrt(2)) * 0.5;
    int rows = compressed.rows;
    int cols = compressed.cols;
    for(int i = 0; i < blockSize; ++i)
    {
        for(int j = 0; j < blockSize; ++j)
        {
            int k = 0, l = 0;
            while(i >> k > 0)
            {
                k++;
            }
            while(j >> l > 0)
            {
                l++;
            }
            int indexi = 0, indexj = 0;
            if(k == l || (k <= 1 && l <= 1))
            {
                if(k > 0)
                {
                    indexi = ((rows/blockSize)<<(k-1)) + i - (1<<(k-1));
                    indexi += ((iblock/blockSize)<<(k-1));
                }
                else
                    indexi = iblock/blockSize;
                if(l > 0)
                {
                    indexj = ((cols/blockSize)<<(l-1)) + j - (1<<(l-1));
                    indexj += ((jblock/blockSize)<<(l-1));
                }
                else
                    indexj = jblock/blockSize;
            }
            else if(k > l)
            {
                indexi = ((rows/blockSize)<<(k-1)) + i - (1<<(k-1));
                indexi += ((iblock/blockSize)<<(k-1));

                indexj = j + ((jblock/blockSize)<<(k-1));
            }
            else if(k < l)
            {

                indexi = i + ((iblock/blockSize)<<(l-1));

                indexj = ((cols/blockSize)<<(l-1)) + j - (1<<(l-1));
                indexj += ((jblock/blockSize)<<(l-1));
            }
            double* compressedRow = compressed.ptr<double>(indexi);
            double* decompressedBlockRow = decompressedBlock.ptr<double>(i);
            decompressedBlockRow[j] = compressedRow[indexj];
        }
    }
    decompressedBlock = haar4.t() * decompressedBlock * haar4;
}
