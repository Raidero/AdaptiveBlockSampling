#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <iostream>
#define MAX_UCHAR 255
using namespace cv;

void compressImage(Mat& source_laplace, Mat& source, std::list<uchar>& output, uchar blockSize);
void blockQuantisation(Mat& roi, Mat& source_roi, std::list<uchar>& compressed);
uchar getMaxColorValue(Mat& roi);
void decompressImage(std::list<uchar>& compressedImage, Mat& decompressedImage);
void blockDequantisation(Mat& roi, std::list<uchar>& compressedImage);
int main( int argc, char** argv )
{
  Mat src, src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  char* window_name = "Laplace Demo";

  /// Load an image
  src = imread( "test.png" );

  if( !src.data )
    { return -1; }

  /// Remove noise by blurring with a Gaussian filter
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Apply Laplace function
  Mat abs_dst;

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );


  //uchar* pointer;
  /*for(int i = 0; i < src.rows; ++i)
  {
    pointer = abs_dst.ptr<uchar>(i);
    for(int j = 200; j < src.cols; ++j)
    {
        pointer[j] = 40;
    }
  }*/
  std::list<uchar> new_image;
  compressImage(abs_dst, src_gray, new_image, 32);
  Mat last_image;
  decompressImage(new_image, last_image);
  /// Show what you got
  imshow( window_name, last_image );
  imwrite("compressed.png", last_image);

  waitKey(0);
  imshow( window_name, src_gray );
  imwrite("gray.png", src_gray);
  waitKey(0);
  return 0;
}

void compressImage(Mat& source_laplace, Mat& source, std::list<uchar>& compressedImage, uchar blockSize)
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
}
