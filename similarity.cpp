/*
 * similarity.cpp
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
#include <vector>
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/xfeatures2d.hpp>  //
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//https://docs.opencv.org/trunk/d5/dc4/tutorial_video_input_psnr_ssim.html
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);        // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

//https://docs.opencv.org/trunk/d5/d6f/tutorial_feature_flann_matcher.html
double getFLANN(const Mat& I1, const Mat& I2)
{
	Mat img_1, img_2;
	cvtColor(I1, img_1, CV_BGR2GRAY);
	cvtColor(I2, img_2, CV_BGR2GRAY);
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
    //-- Step 2: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	double max_dist = 0, min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for(int i = 0; i < descriptors_1.rows; i++)
	{ 
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	return 0; // decide what to return 
}

//https://docs.opencv.org/trunk/d5/dde/tutorial_feature_description.html
double getBFMatch(const Mat& I1, const Mat& I2)
{
	Mat img_1, img_2;
	cvtColor(I1, img_1, CV_BGR2GRAY);
	cvtColor(I2, img_2, CV_BGR2GRAY);
	 //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1);
	detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2);
	//-- Step 2: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	return 0; // decide what to return
}

//https://docs.opencv.org/trunk/d8/dc8/tutorial_histogram_comparison.html
double getHistMatch(const Mat& src_base, const Mat& src_test)
{
	Mat hsv_base, hsv_test;
	cvtColor(src_base, hsv_base, COLOR_BGR2HSV);
	cvtColor(src_test, hsv_test, COLOR_BGR2HSV);
	// Initialize the arguments to calculate the histograms (bins, ranges and channels H and S) 
	int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins};
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180};
    float s_ranges[] = { 0, 256};
    const float* ranges[] = { h_ranges, s_ranges};
	int channels[] = { 0, 1};
	// Create the MatND objects to store the histograms
	MatND hist_base, hist_test;
	// Calculate the histogram for base and test image
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
	calcHist(&hsv_test, 1, channels, Mat(), hist_test, 2, histSize, ranges, true, false);
	normalize(hist_test, hist_test, 0, 1, NORM_MINMAX, -1, Mat());
	// Apply comparison
	double base_test = compareHist(hist_base, hist_test, HISTCMP_CORREL);
	return base_test;
}

//https://docs.opencv.org/trunk/d5/dc4/tutorial_video_input_psnr_ssim.html
Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}

int main(int argc, char **argv)
{
	Mat im1 = imread("./a.jpg", -1); 
    Mat im2 = imread("./b.jpg", -1);
    resize(im2, im2, im1.size(), 0, 0);
    printf("%f\n", getPSNR(im1, im2));
    printf("%f\n", getHistMatch(im1, im2));
    Scalar MSSIM = getMSSIM(im1, im2);
    cout << "M = "<< endl << " " <<  MSSIM << endl << endl;
    // getBFMatch(im1, im2);
    // getFLANN(im1, im2);
   	return 0;
}

