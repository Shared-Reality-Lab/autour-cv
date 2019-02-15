/*
 * blur.cpp
 * 
 * Copyright 2018 s225055 <s225055@s225055>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/xfeatures2d.hpp>  //
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect.hpp>
using namespace std;
using namespace cv;

//https://stackoverflow.com/a/20198278/7025019
double CannyThreshold(const Mat& src)
{
	Mat src_gray, detected_edges;
	int lowThreshold = 30, ratio = 3, kernel_size = 3;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3,3));
	Canny(src, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	int non_zero = countNonZero(detected_edges); double d = (double) non_zero;
	return d/(detected_edges.rows*detected_edges.cols);
 }

//https://stackoverflow.com/a/7768918/7025019 
double modifiedLaplacian(const Mat& src)
{
    Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
    Mat G = getGaussianKernel(3, -1, CV_64F);
    Mat Lx, Ly;
    sepFilter2D(src, Lx, CV_64F, M, G);
    sepFilter2D(src, Ly, CV_64F, G, M);
    Mat FM = abs(Lx)+abs(Ly);
    double focusMeasure = mean(FM).val[0];
    return focusMeasure;
}

double varianceOfLaplacian(const cv::Mat& src)
{
	Mat lap;
    Laplacian(src, lap, CV_64F);
    Scalar mu, sigma;
    meanStdDev(lap, mu, sigma);
    double focusMeasure = sigma.val[0]*sigma.val[0];
    return focusMeasure;
}

double normalizedGraylevelVariance(const Mat& src)
{
    Scalar mu, sigma;
    meanStdDev(src, mu, sigma);
    double focusMeasure = (sigma.val[0]*sigma.val[0])/mu.val[0];
    return focusMeasure;
}

int main(int argc, char **argv)
{
	Mat im1 = imread("a.jpeg", -1);
	printf("%f\n", CannyThreshold(im1));
	printf("%f\n", modifiedLaplacian(im1));
	printf("%f\n", normalizedGraylevelVariance(im1));
	printf("%f\n", varianceOfLaplacian(im1));
	return 0;
}

