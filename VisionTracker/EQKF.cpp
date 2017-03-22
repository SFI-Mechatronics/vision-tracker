#include "EQKF.hpp"

// Constructor
EQKF::EQKF()
{
	// Call init function
	InitH();
}

// Functions
void EQKF::InitH()
{
	// Define size of private variables
	F = Eigen::MatrixXd::Zero(19, 19);
	H = Eigen::MatrixXd::Zero(7, 19);
	HT = Eigen::MatrixXd::Zero(19, 7);
	P = Eigen::MatrixXd::Zero(19, 19);
	PPred = Eigen::MatrixXd::Zero(19, 19);
	Q = Eigen::MatrixXd::Identity(19, 19);
	R = Eigen::MatrixXd::Identity(7, 7);
	x = Eigen::VectorXd::Zero(19, 1);
	z = Eigen::VectorXd::Zero(7, 1);

	// Initialize state vectors
	x(9, 0) = 1.0;
	xOld = x;
	xPred = x;
	z(3, 0) = x(9, 0);
	qOld = Eigen::Matrix3d::Identity(3, 3);

	// Define and transpose measurement matrix
	H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(3, 3);
	H.block<4, 4>(3, 9) = Eigen::Matrix4d::Identity(4, 4);
	HT = H.transpose();

	// Default params
	dt = 0.075;
	q = 0.05;
	r = 0.05;
}

void EQKF::Predict()
{
	double q0, q1, q2, q3, w1, w2, w3, w1_t, w2_t, w3_t;
	
	// Change state names
	q0 = xOld(9, 0);
	q1 = xOld(10, 0);
	q2 = xOld(11, 0);
	q3 = xOld(12, 0);
	
	w1 = xOld(13, 0);
	w2 = xOld(14, 0);
	w3 = xOld(15, 0);

	w1_t = xOld(16, 0);
	w2_t = xOld(17, 0);
	w3_t = xOld(18, 0);
	
	// Linearized system matrix K
	F(0, 0) = 1.0;
	F(0, 3) = dt;
	F(0, 6) = (dt*dt)*(1.0 / 2.0);
	F(1, 1) = 1.0;
	F(1, 4) = dt;
	F(1, 7) = (dt*dt)*(1.0 / 2.0);
	F(2, 2) = 1.0;
	F(2, 5) = dt;
	F(2, 8) = (dt*dt)*(1.0 / 2.0);
	F(3, 3) = 1.0;
	F(3, 6) = dt;
	F(4, 4) = 1.0;
	F(4, 7) = dt;
	F(5, 5) = 1.0;
	F(5, 8) = dt;
	F(6, 6) = 1.0;
	F(7, 7) = 1.0;
	F(8, 8) = 1.0;
	F(9, 9) = dt*(q0*q1*(w1 + dt*w1_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) + q0*q2*(w2 + dt*w2_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) + q0*q3*(w3 + dt*w3_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0)) + 1.0;
	F(9, 10) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w1 + (q2*q2)*w1 + (q3*q3)*w1 + dt*(q0*q0)*w1_t + dt*(q2*q2)*w1_t + dt*(q3*q3)*w1_t - q1*q2*w2 - q1*q3*w3 - dt*q1*q2*w2_t - dt*q1*q3*w3_t)*(-1.0 / 2.0);
	F(9, 11) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w2 + (q1*q1)*w2 + (q3*q3)*w2 + dt*(q0*q0)*w2_t + dt*(q1*q1)*w2_t + dt*(q3*q3)*w2_t - q1*q2*w1 - q2*q3*w3 - dt*q1*q2*w1_t - dt*q2*q3*w3_t)*(-1.0 / 2.0);
	F(9, 12) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w3 + (q1*q1)*w3 + (q2*q2)*w3 + dt*(q0*q0)*w3_t + dt*(q1*q1)*w3_t + dt*(q2*q2)*w3_t - q1*q3*w1 - q2*q3*w2 - dt*q1*q3*w1_t - dt*q2*q3*w2_t)*(-1.0 / 2.0);
	F(9, 13) = dt*q1*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(9, 14) = dt*q2*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(9, 15) = dt*q3*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(9, 16) = (dt*dt)*q1*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(9, 17) = (dt*dt)*q2*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(9, 18) = (dt*dt)*q3*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(10, 9) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q1*q1)*w1 + (q2*q2)*w1 + (q3*q3)*w1 + dt*(q1*q1)*w1_t + dt*(q2*q2)*w1_t + dt*(q3*q3)*w1_t + q0*q2*w3 - q0*q3*w2 + dt*q0*q2*w3_t - dt*q0*q3*w2_t)*(1.0 / 2.0);
	F(10, 10) = -dt*(q0*q1*(w1 + dt*w1_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) + q1*q3*(w2 + dt*w2_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) - q1*q2*(w3 + dt*w3_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0)) + 1.0;
	F(10, 11) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w3 + (q1*q1)*w3 + (q3*q3)*w3 + dt*(q0*q0)*w3_t + dt*(q1*q1)*w3_t + dt*(q3*q3)*w3_t + q0*q2*w1 + q2*q3*w2 + dt*q0*q2*w1_t + dt*q2*q3*w2_t)*(-1.0 / 2.0);
	F(10, 12) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w2 + (q1*q1)*w2 + (q2*q2)*w2 + dt*(q0*q0)*w2_t + dt*(q1*q1)*w2_t + dt*(q2*q2)*w2_t - q0*q3*w1 + q2*q3*w3 - dt*q0*q3*w1_t + dt*q2*q3*w3_t)*(1.0 / 2.0);
	F(10, 13) = dt*q0*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(10, 14) = dt*q3*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(10, 15) = dt*q2*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(10, 16) = (dt*dt)*q0*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(10, 17) = (dt*dt)*q3*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(10, 18) = (dt*dt)*q2*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(11, 9) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q1*q1)*w2 + (q2*q2)*w2 + (q3*q3)*w2 + dt*(q1*q1)*w2_t + dt*(q2*q2)*w2_t + dt*(q3*q3)*w2_t - q0*q1*w3 + q0*q3*w1 - dt*q0*q1*w3_t + dt*q0*q3*w1_t)*(1.0 / 2.0);
	F(11, 10) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w3 + (q2*q2)*w3 + (q3*q3)*w3 + dt*(q0*q0)*w3_t + dt*(q2*q2)*w3_t + dt*(q3*q3)*w3_t - q0*q1*w2 + q1*q3*w1 - dt*q0*q1*w2_t + dt*q1*q3*w1_t)*(1.0 / 2.0);
	F(11, 11) = -dt*(q0*q2*(w2 + dt*w2_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) - q2*q3*(w1 + dt*w1_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) + q1*q2*(w3 + dt*w3_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0)) + 1.0;
	F(11, 12) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w1 + (q1*q1)*w1 + (q2*q2)*w1 + dt*(q0*q0)*w1_t + dt*(q1*q1)*w1_t + dt*(q2*q2)*w1_t + q0*q3*w2 + q1*q3*w3 + dt*q0*q3*w2_t + dt*q1*q3*w3_t)*(-1.0 / 2.0);
	F(11, 13) = dt*q3*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(11, 14) = dt*q0*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(11, 15) = dt*q1*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(11, 16) = (dt*dt)*q3*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(11, 17) = (dt*dt)*q0*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(11, 18) = (dt*dt)*q1*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(12, 9) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q1*q1)*w3 + (q2*q2)*w3 + (q3*q3)*w3 + dt*(q1*q1)*w3_t + dt*(q2*q2)*w3_t + dt*(q3*q3)*w3_t + q0*q1*w2 - q0*q2*w1 + dt*q0*q1*w2_t - dt*q0*q2*w1_t)*(1.0 / 2.0);
	F(12, 10) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w2 + (q2*q2)*w2 + (q3*q3)*w2 + dt*(q0*q0)*w2_t + dt*(q2*q2)*w2_t + dt*(q3*q3)*w2_t + q0*q1*w3 + q1*q2*w1 + dt*q0*q1*w3_t + dt*q1*q2*w1_t)*(-1.0 / 2.0);
	F(12, 11) = dt*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*((q0*q0)*w1 + (q1*q1)*w1 + (q3*q3)*w1 + dt*(q0*q0)*w1_t + dt*(q1*q1)*w1_t + dt*(q3*q3)*w1_t - q0*q2*w3 + q1*q2*w2 - dt*q0*q2*w3_t + dt*q1*q2*w2_t)*(1.0 / 2.0);
	F(12, 12) = -dt*(q2*q3*(w1 + dt*w1_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) - q1*q3*(w2 + dt*w2_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0) + q0*q3*(w3 + dt*w3_t)*1.0 / pow(q0*q0 + q1*q1 + q2*q2 + q3*q3, 3.0 / 2.0)*(1.0 / 2.0)) + 1.0;
	F(12, 13) = dt*q2*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(12, 14) = dt*q1*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(12, 15) = dt*q0*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(12, 16) = (dt*dt)*q2*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(12, 17) = (dt*dt)*q1*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(-1.0 / 2.0);
	F(12, 18) = (dt*dt)*q0*1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)*(1.0 / 2.0);
	F(13, 13) = 1.0;
	F(13, 16) = dt;
	F(14, 14) = 1.0;
	F(14, 17) = dt;
	F(15, 15) = 1.0;
	F(15, 18) = dt;
	F(16, 16) = 1.0;
	F(17, 17) = 1.0;
	F(18, 18) = 1.0;
	
	// Predicted state estimate
	xPred = F*xOld;

	// Predicted covariance
	PPred = F*P*F.transpose() + q*q*Q;
}

Eigen::Matrix4d EQKF::Correct(Eigen::Matrix4d M)
{
	Eigen::MatrixXd K, S;
	Eigen::Matrix3d R_est;
	Eigen::Matrix4d M_est;
	Eigen::Quaterniond qAvg, qEst;
	Eigen::Vector3d posAvg;

	// Innovation covariance
	S = H*PPred*HT + r*r*R;

	// Near-optimal kalman gain
	K = PPred*HT*S.inverse();
	
	// Correct state estimate
	posAvg = M.block<3, 1>(0, 3);
	qAvg = M.block<3, 3>(0, 0);
	qAvg.normalize();

	z(0, 0) = posAvg(0, 0);
	z(1, 0) = posAvg(1, 0);
	z(2, 0) = posAvg(2, 0);

	//qAvg = CheckQuartonian(qAvg);

	z(3, 0) = qAvg.w();
	z(4, 0) = qAvg.x();
	z(5, 0) = qAvg.y();
	z(6, 0) = qAvg.z();


	x = xPred + K*(z - H*xPred);
	
	// Return estimated H matrix
	M_est = Eigen::Matrix4d::Identity(4, 4);
	M_est.block<3, 1>(0, 3) = x.block<3, 1>(0, 0);
	qEst.w() = x(9, 0);
	qEst.x() = x(10, 0);
	qEst.y() = x(11, 0);
	qEst.z() = x(12, 0);
	qEst.normalize();

	R_est = qEst.toRotationMatrix();
	M_est.block<3, 3>(0, 0) = R_est;


	// Update covariance estimate
	P = (Eigen::MatrixXd::Zero(19, 19) - K*H)*PPred;

	// Update xOld;
	xOld = x;

	// Return estimated H matrix
	return M_est;
}


