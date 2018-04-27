#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <iostream>
#include <math.h>
using namespace cv;

void compress(Mat_<double>& input, Mat_<double>& output, short blockSize);
void compressBlock(Mat_<double>& input, Mat_<double>& output, short blockSize, int iblock, int jblock);
void decompress(Mat_<double>& input, Mat_<double>& output, short blockSize);
void decompressBlock(Mat_<double>& input, Mat_<double>& output, short blockSize, int iblock, int jblock);

int main( int argc, char** argv )
{
    Mat src, src_gray;
    std::string window_name = "Rip";

    src = imread( "test3.png" );
    Mat_<double> src_double;
    cvtColor( src, src_gray, CV_BGR2GRAY );
    src_gray.convertTo(src_double, CV_64FC1);
    if( !src.data )
        return -1;

    Mat_<double> compressed = src_double.clone();
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );


    compress(src_double, compressed, 4);
    /*double min, max;
    cv::minMaxLoc(compressed, &min, &max);
    compressed -= min;
    compressed /= (max - min);
    compressed *= 256;*/
    compressed.convertTo(src_gray, CV_8UC1);
    imwrite("compressed.png", src_gray);
    imshow(window_name, src_gray);
    waitKey();


    decompress(compressed, src_double, 4);
    src_double.convertTo(src_gray, CV_8UC1);
    imshow( window_name, src_gray );
    imwrite("decompressed.png", src_gray);

    waitKey(0);

    return 0;
}

void compress(Mat_<double>& input, Mat_<double>& output, short blockSize)
{
    for(int i = 0; i < input.rows; i += blockSize)
    {
        for(int j = 0; j < input.cols; j += blockSize)
        {
            Mat_<double> block = input(Rect2d(j, i, blockSize, blockSize));
            compressBlock(block, output, blockSize, i, j);
        }
    }
}



void compressBlock(Mat_<double>& input, Mat_<double>& output, short blockSize, int iblock, int jblock)
{
    static Mat_<double> haar8 = (Mat_<double>(8,8) << 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, -1, -1, -1, -1,
                                    2, 2, -2, -2, 0, 0, 0, 0,
                                    0, 0, 0, 0, 2, 2, -2, -2,
                                    4, -4, 0, 0, 0, 0, 0, 0,
                                    0, 0, 4, -4, 0, 0, 0, 0,
                                    0, 0, 0, 0, 4, -4, 0, 0,
                                    0, 0, 0, 0, 0, 0, 4, -4) / 8.0;
    static Mat_<double> haar4 = (Mat_<double>(4,4) << 1, 1, 1, 1,
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

void decompress(Mat_<double>& compressed, Mat_<double>& decompressed, short blockSize)
{
    for(int i = 0; i < compressed.rows; i += blockSize)
    {
        for(int j = 0; j < compressed.cols; j += blockSize)
        {
            Mat_<double> block = decompressed(Rect2d(j, i, blockSize, blockSize));
            decompressBlock(block, compressed, blockSize, i, j);
        }
    }
}

void decompressBlock(Mat_<double>& decompressedBlock, Mat_<double>& compressed, short blockSize, int iblock, int jblock)
{
    static Mat_<double> haar8 = (Mat_<double>(8,8) << 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, -1, -1, -1, -1,
                                    2, 2, -2, -2, 0, 0, 0, 0,
                                    0, 0, 0, 0, 2, 2, -2, -2,
                                    4, -4, 0, 0, 0, 0, 0, 0,
                                    0, 0, 4, -4, 0, 0, 0, 0,
                                    0, 0, 0, 0, 4, -4, 0, 0,
                                    0, 0, 0, 0, 0, 0, 4, -4) / 8.0;
    static Mat_<double> haar4 = (Mat_<double>(4,4) << 1, 1, 1, 1,
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
/*void compressImage(Mat& source_laplace, Mat& source, std::list<uchar>& compressedImage, uchar blockSize)
{
    //compressedImage.push_back(source_laplace.rows);
    //compressedImage.push_back(source_laplace.cols);
    compressedImage.push_back(blockSize);
    uchar row, col;
    for(int i = 0; i < source_laplace.rows; i += blockSize)
    {
        for(int j = 0; j < source_laplace.cols; j += blockSize)
        {
            if(source_laplace.rows - i < blockSize)
                row = source_laplace.rows - i;
            else
                row = blockSize;
            if(source_laplace.cols - j < blockSize)
                col = source_laplace.cols - j;
            else
                col = blockSize;
            Mat roi = source_laplace(Rect(j, i, col, row));
            Mat roi2 = source(Rect(j, i, col, row));
            blockQuantisation(roi, roi2, compressedImage);
        }
    }
}

void blockQuantisation(Mat& roi, Mat& source_roi, std::list<uchar>& compressed)
{
    uchar period = MAX_UCHAR / getMaxColorValue(roi);
    compressed.push_back(period);
    compressed.push_back(roi.rows);
    compressed.push_back(roi.cols);

    uchar* row;
    for(int i = 0; i < roi.rows; i += period)
    {
        row = source_roi.ptr<uchar>(i);
        for(int j = 0; j < roi.cols; j += period)
        {
            compressed.push_back(row[j]);
        }
    }
}

uchar getMaxColorValue(Mat& roi)
{
    uchar* row;
    uchar maxColorValue = 0;
    for(int i = 0; i < roi.rows; ++i)
    {
        row = roi.ptr<uchar>(i);
        for(int j = 0; j < roi.cols; ++j)
        {
            if(row[j] > maxColorValue)
            {
                maxColorValue = row[j];
            }
        }
    }
    return maxColorValue;
}

void decompressImage(std::list<uchar>& compressedImage, Mat& decompressedImage)
{
    uchar row, col;
    int rows = 768;//compressedImage.front();
    //compressedImage.pop_front();
    int cols = 1024;//compressedImage.front();
    //compressedImage.pop_front();
    uchar blockSize = compressedImage.front();
    compressedImage.pop_front();
    decompressedImage = Mat::zeros(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i += blockSize)
    {
        for(int j = 0; j < cols; j += blockSize)
        {
            if(decompressedImage.rows - i < blockSize)
                row = decompressedImage.rows - i;
            else
                row = blockSize;
            if(decompressedImage.cols - j < blockSize)
                col = decompressedImage.cols - j;
            else
                col = blockSize;
            Mat roi = decompressedImage(Rect(j, i, col, row));
            blockDequantisation(roi, compressedImage);
        }
    }
}

void blockDequantisation(Mat& roi, std::list<uchar>& compressedImage)
{
    uchar period = compressedImage.front();
    compressedImage.pop_front();
    uchar rows = compressedImage.front();
    compressedImage.pop_front();
    uchar cols = compressedImage.front();
    compressedImage.pop_front();

    uchar* row;
    for(int i = 0; i < rows; i += period)
    {
        row = roi.ptr<uchar>(i);
        for(int j = 0; j < cols; j += period)
        {
            row[j] = compressedImage.front();
            compressedImage.pop_front();
        }
    }
}*/
