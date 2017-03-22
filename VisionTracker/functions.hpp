#pragma once
// Standard includes
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <ctime>
#include <winsock2.h>
#include <stdio.h>
#include <Windows.h>
#include <string>
#include <vector>

// OpenCV includes
#include "opencv/cv.h"
#include "opencv2/aruco.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/StdVector>


#define WIN32_LEAN_AND_MEAN
#pragma comment(lib,"ws2_32.lib") //Winsock Library

// Define UDP comm properties
//#define SERVER "127.0.0.1"  //ip address of udp client
//#define PORT 8999   //The port on which to listen for incoming data
#define SERVER "192.168.90.180"  //ip address of udp client
#define PORT 50035 //The port on which to listen for incoming data


// Structs
struct UdpDataRead
{
	int test = 0;
	
	float q[6];
	float posSP1[6];
	float posSP2[6];
};

struct UdpDataSend
{
	int dataReady = 0;

	float posAvg[7];
	float posEst[7];
};

// Functions
static void WriteMatToCSV(Eigen::Matrix4d H, std::ofstream &file)
{
	// Write matrix H as CSV string in file object
	for (unsigned int j = 0; j < 4; j++)
	{
		for (unsigned int i = 0; i < 4; i++)
		{
			if ((i == 3) && (j == 3))
			{
				file << H(i, j) << "\n";
			}
			else
			{
				file << H(i, j) << ",";
			}
		}
	}
}

static void PrintManual()
{
	std::cout << "****** USER MANUAL ******" << std::endl;
	std::cout << "C     : Save Aruco side offsets" << std::endl;
	std::cout << "SPACE : Save snapshot from camera" << std::endl;
	std::cout << "ENTER : Save Hand-Eye data" << std::endl;
	std::cout << "ESC   : Exit" << std::endl;
}

static double AreaQuadrilateral(std::vector<cv::Point2f> corners)
{
	// Source: https://en.wikipedia.org/wiki/Shoelace_formula
	double x1, x2, x3, x4;
	double y1, y2, y3, y4;
	double A = 0;

	x1 = corners[0].x;
	x2 = corners[1].x;
	x3 = corners[2].x;
	x4 = corners[3].x;

	y1 = corners[0].y;
	y2 = corners[1].y;
	y3 = corners[2].y;
	y4 = corners[3].y;

	A = abs(0.5*(x1*y2 + x2*y3 + x3*y4 + x4*y1 - x2*y1 - x3*y2 - x4*y3 - x1*y4));

	return A;
}

namespace Eigen {
	template<class Matrix>
	void write_binary(const char* filename, const Matrix& matrix) {
		std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
		typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
		out.write((char*)(&rows), sizeof(typename Matrix::Index));
		out.write((char*)(&cols), sizeof(typename Matrix::Index));
		out.write((char*)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
		out.close();
	}
	template<class Matrix>
	void read_binary(const char* filename, Matrix& matrix) {
		std::ifstream in(filename, std::ios::in | std::ios::binary);
		typename Matrix::Index rows = 0, cols = 0;
		in.read((char*)(&rows), sizeof(typename Matrix::Index));
		in.read((char*)(&cols), sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.read((char *)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
		in.close();
	}
}

static Eigen::Matrix4d VecsToMat(cv::Vec3d rvec, cv::Vec3d tvec)
{
	Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
	cv::Mat R;

	// Reallocate from rvec to H
	cv::Rodrigues(rvec, R);

	for (unsigned int m = 0; m < 3; m++)
	{
		for (unsigned int n = 0; n < 3; n++)
		{
			H(m, n) = R.at<double>(m, n);

		}
	}

	// Reallocate from tvec to H
	for (unsigned int m = 0; m < 3; m++)
	{
		H(m, 3) = tvec(m);
	}

	return H;
}

static void MatToVecs(cv::Vec3d &rvec, cv::Vec3d &tvec, Eigen::Matrix4d H)
{
	cv::Vec3d temp1, temp2;
	cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

	
	for (unsigned int m = 0; m < 3; m++)
	{
		for (unsigned int n = 0; n < 3; n++)
		{
			R.at<double>(m, n) = H(m, n);
		}
	}

	
	// Reallocate from H to rvec
	cv::Rodrigues(R, temp1);

	

	// Reallocate from tvec to H
	for (unsigned int m = 0; m < 3; m++)
	{
		temp2(m) = H(m, 3);
	}

	// Return tvec and rvec
	rvec = temp1;
	tvec = temp2;
}

static void DrawAxis(cv::InputOutputArray image, Eigen::Matrix4d H, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, float markerLength)
{
	cv::Vec3d rvec, tvec;
	MatToVecs(rvec, tvec, H);

	cv::aruco::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, markerLength);
}




static Eigen::Matrix4d TransDH(double a, double alpha, double d, double theta)
{
	// Denavit-Hartenberg Transformation matrix
	Eigen::Matrix4d T = Eigen::Matrix<double, 4, 4>::Identity();

	T(0, 0) = cos(theta);
	T(0, 1) = -cos(alpha)*sin(theta);
	T(0, 2) = sin(alpha)*sin(theta);
	T(0, 3) = a*cos(theta);
	T(1, 0) = sin(theta);
	T(1, 1) = cos(alpha)*cos(theta);
	T(1, 2) = -sin(alpha)*cos(theta);
	T(1, 3) = a*sin(theta);
	T(2, 1) = sin(alpha);
	T(2, 2) = cos(alpha);
	T(2, 3) = d;
	T(3, 3) = 1.0;

	return T;
}

static Eigen::Matrix4d ComauFK(float q[6])
{
	// Matrices define
	Eigen::Matrix4d A1(4, 4), A2(4, 4), A3(4, 4), A4(4, 4), A5(4, 4), A6(4, 4);
	Eigen::Matrix4d T06;

	// Comau S5 NJ-110 3.0 geomtric parameters
	double a1 = 0.350;
	double a2 = 1.160;
	double a3 = 0.250;
	double d1 = 0.830;
	double d4 = 1.4922;
	double d6 = 0.210;

	// Convert from deg to rad
	for (unsigned int i = 0; i < 6; i++)
	{
		q[i] = q[i] / 180.0 * M_PI;
	}

	// Convert Comau angles to serial angles
	q[0] = -q[0];
	q[2] = q[2] + q[1] + M_PI / 2.0;
	q[1] = -q[1] + M_PI / 2.0;
	q[3] = -q[3];
	q[4] = -q[4];
	q[5] = -q[5];

	// DH transformations for each single link
	A1 = TransDH(a1, M_PI / 2.0, d1, q[0]);
	A2 = TransDH(a2, 0.0, 0.0, q[1]);
	A3 = TransDH(a3, M_PI / 2.0, 0.0, q[2]);
	A4 = TransDH(0.0, -M_PI / 2.0, d4, q[3]);
	A5 = TransDH(0.0, M_PI / 2.0, 0.0, q[4]);
	A6 = TransDH(0.0, 0.0, d6, q[5] + M_PI);

	// Calculate robot pose and position matrix
	T06 = A1*A2*A3*A4*A5*A6;

	// Return robot pose and position matrix
	return T06;
}

static Eigen::Matrix4d GetPoseSP(float pos[6])
{
	Eigen::Matrix4d H;

	H(0, 0) = cos(pos[4])*cos(pos[5]);
	H(0, 1) = -cos(pos[4])*sin(pos[5]);
	H(0, 2) = sin(pos[4]);
	H(0, 3) = pos[0];
	H(1, 0) = cos(pos[3])*sin(pos[5]) + cos(pos[5])*sin(pos[4])*sin(pos[3]);
	H(1, 1) = cos(pos[3])*cos(pos[5]) - sin(pos[4])*sin(pos[3])*sin(pos[5]);
	H(1, 2) = -cos(pos[4])*sin(pos[3]);
	H(1, 3) = pos[1];
	H(2, 0) = sin(pos[3])*sin(pos[5]) - cos(pos[3])*cos(pos[5])*sin(pos[4]);
	H(2, 1) = cos(pos[5])*sin(pos[3]) + cos(pos[3])*sin(pos[4])*sin(pos[5]);
	H(2, 2) = cos(pos[4])*cos(pos[3]);
	H(2, 3) = pos[2];
	H(3, 3) = 1.0;

	return H;
}

/*
static bool readDetectorParameters(std::string filename, cv::Ptr<cv::aruco::DetectorParameters> &params)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["doCornerRefinement"] >> params->doCornerRefinement;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

static bool readCameraParameters(std::string filename, cv::Mat &camMatrix, cv::Mat &distCoeffs)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}
*/