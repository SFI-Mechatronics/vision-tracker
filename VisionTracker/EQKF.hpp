#pragma once

#include <iostream>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

class EQKF
{
public:
	// Variables
	double dt;
	double q;
	double r;

	// Constructor
	EQKF::EQKF();

	// Functions
	void Predict();
	Eigen::Matrix4d Correct(Eigen::Matrix4d M);
private:
	// Variables
	Eigen::VectorXd x, xPred, xOld, z;
	Eigen::MatrixXd F, H, HT, P, PPred, Q, R;
	Eigen::Quaterniond qOld;
	
	// Functions
	void InitH();
	bool BoolSign(double u);
	Eigen::Quaterniond CheckQuartonian(Eigen::Quaterniond qIn);
};