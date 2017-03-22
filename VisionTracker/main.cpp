// Include libraries and functions
#include "functions.hpp"
#include "EQKF.hpp"

// VisionTracker Notes
// Sondre Sanden Tørdal
// Averaging quartenions: http://stackoverflow.com/questions/12374087/average-of-multiple-quaternions

// Define desired marker ids
#define MARKER_ID_1 1
#define MARKER_ID_2 2
#define MARKER_ID_3 3
#define MARKER_ID_4 4
#define MARKER_ID_5 5
#define MARKER_ID_6 6

// Define program modes
#define STOP			0
#define CALIB_MARKERS	1
#define RUN				3

UdpDataRead* udpRead;
UdpDataSend udpSend;
bool closeUdp = false;
bool udpActive = false;

void UdpCommunication()
{
	int s, slen;
	struct sockaddr_in server, si_other;
	WSADATA wsa;

	slen = sizeof(si_other);

	// Initialise winsock
	printf("\nInitialising Winsock...");
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		printf("Failed. Error Code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
	}
	printf("Initialised.\n");

	// Create a socket
	if ((s = static_cast<int>(socket(AF_INET, SOCK_DGRAM, 0))) == INVALID_SOCKET)
	{
		printf("Could not create socket : %d", WSAGetLastError());
	}
	printf("Socket created.\n");

	// Prepare the sockaddr_in structure
	server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons(PORT);

	// Bind to socket
	if (bind(s, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR)
	{
		printf("Bind failed with error code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
	}
	puts("Bind done");

	// Initilize udp buffers
	char udpBufferRead[sizeof(UdpDataRead)];
	char udpBufferWrite[sizeof(UdpDataSend)];

	// Keep listening for data
	while (true)
	{	
		// Try to receive some data, this is a blocking call
		if ((recvfrom(s, udpBufferRead, sizeof(udpBufferRead), 0, (struct sockaddr*) &si_other, &slen)) == SOCKET_ERROR)
		{
			printf("recvfrom() failed with error code : %d", WSAGetLastError());
			exit(EXIT_FAILURE);
		}
		else
		{
			// Set UDP active status
			udpActive = true;

			// Cast recieved buffer to struct
			udpRead = (UdpDataRead*)udpBufferRead;

			// Cast struct to send buffer
			memcpy(&udpBufferWrite, &udpSend, sizeof(udpSend));
		}
		
		// Reply the client with data
		if (sendto(s, udpBufferWrite, sizeof(udpBufferWrite), 0, (struct sockaddr*) &si_other, slen) == SOCKET_ERROR)
		{
			printf("sendto() failed with error code : %d", WSAGetLastError());
			exit(EXIT_FAILURE);
		}
	
		// Close thread based on condition
		if (closeUdp)
		{
			udpActive = false;
			break;
		}
	}
	std::cout << "udpThred exited" << std::endl;
	closesocket(s);
	WSACleanup();

	return;
}


int main(int argc, char** argv)
{
	// Program mode
	int mode = RUN;

	// Start UDP communication thread
	std::thread udpThread(UdpCommunication);
	
	// General settings
	cv::setUseOptimized(true);
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	// Describe the intrinsic and distortion coefficients
	cv::Mat camMatrix, distCoeffs;

	// Intrinsic parameters from scaled (camScale = 0.5) picture
	const double camScale = 0.5;
	camMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
	camMatrix.at<double>(0, 0) = 582.64590; // fx
	camMatrix.at<double>(1, 1) = 581.19390; // fy
	camMatrix.at<double>(2, 2) = 1.0;
	camMatrix.at<double>(0, 2) = 457.91989; // cx
	camMatrix.at<double>(1, 2) = 274.12724; // cy

	// Distortion = [k1 k2 p1 p2 k3]
	distCoeffs = cv::Mat::zeros(5, 1, CV_64FC1);
	distCoeffs.at<double>(0, 0) = 0.05394; // k1
	distCoeffs.at<double>(1, 0) = -0.12348; // k2

	// Create aruco markers
	const int markerSize = 2000;
	const int numberOfMarkers = 6;
	int markerId[numberOfMarkers] = {MARKER_ID_1, MARKER_ID_2, MARKER_ID_3, MARKER_ID_4, MARKER_ID_5, MARKER_ID_6};
	bool markerFound[numberOfMarkers];
	std::string markerName;
	cv::Mat markerImage;

	// Define the aruco dictionary size
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

	// Save aruco markers to .png images
	for (int i = 0; i < numberOfMarkers; i++)
	{
		markerName = "./aruco_markers/markerId_" + std::to_string(markerId[i]) + ".png";
		cv::aruco::drawMarker(dictionary, markerId[i], markerSize, markerImage, 1);
		cv::imwrite(markerName, markerImage);
	}

	// Specify the physical marker length
	float markerLength = 0.29f; // m Comment: tvecs have the same output units
	
	// Define variables
	int cubeCalibCount[4] = { 0, 0, 0, 0 };
	int saveImageCount = 0;
	int handEyeCount = 1;
	std::string saveName;

	// Image storing matrices
	cv::Mat image, imageCopy;

	// Webcam properties
	cv::VideoCapture cam;
	cam.open(0);
	cam.set(CV_CAP_PROP_FRAME_WIDTH, (int)(1920.0 * camScale));
	cam.set(CV_CAP_PROP_FRAME_HEIGHT, (int)(1080.0 * camScale));

	// Aruco marker processing storage
	std::vector < int > ids;
	std::vector < std::vector < cv::Point2f > > corners, rejected;
	std::vector < cv::Vec3d > rvecs, tvecs;

	// Calibration data files
	std::ofstream fileCAM, fileROB, fileH21, fileH31, fileH41, fileH51;
	fileCAM.open("./hand_eye_calib/dataCAM.txt");
	fileROB.open("./hand_eye_calib/dataROB.txt");
	fileH21.open("./calib_matrices/dataH21.txt");
	fileH31.open("./calib_matrices/dataH31.txt");
	fileH41.open("./calib_matrices/dataH41.txt");
	fileH51.open("./calib_matrices/dataH51.txt");

	// Augumented reality markers
	cv::Point principalPoint = {(int)camMatrix.at<double>(0, 2), (int)camMatrix.at<double>(1, 2)};

	// Homogenous transformation matrices
	Eigen::Matrix4d M1, M2, M3, M4, M5, M6;
	Eigen::Matrix4d T1, T2, T3, T4, T5, T6;
	Eigen::Matrix4d H21, H31, H41, H51, H10;
	Eigen::Matrix4d H, H_est;
	Eigen::Quaterniond q1, q2, q3, q4, q5, q6, qAvg, qEst;
	double a1, a2, a3, a4, a5, a6;

	H = Eigen::MatrixXd::Identity(4, 4);
	H_est = Eigen::MatrixXd::Identity(4, 4);

	// Cube side calibration matrices from matlab
	H21.row(0) << -0.0031, -0.0024, -1.0000, -0.1964;
	H21.row(1) << 0.9998, 0.0172, -0.0031, 0.0001;
	H21.row(2) << 0.0172, -0.9998, 0.0024, -0.1973;
	H21.row(3) << 0, 0, 0, 1;
	
	H31.row(0) << 0.0045, -0.0016, -1.0000, -0.1967;
	H31.row(1) << 0.0114, -0.9999, 0.0017, 0.0021;
	H31.row(2) << -0.9999, -0.0114, -0.0044, -0.1987;
	H31.row(3) << 0, 0, 0, 1;

	H41.row(0) << -0.9999, -0.0132, -0.0021, 0.0021;
	H41.row(1) << -0.0020, -0.0031, 1.0000, 0.1969;
	H41.row(2) << -0.0132, 0.9999, 0.0030, -0.1981;
	H41.row(3) << 0, 0, 0, 1;

	H51.row(0) << -0.0123, 0.9999, 0.0004, 0.0000;
	H51.row(1) << 0.0006, -0.0003, 1.0000, 0.1963;
	H51.row(2) << 0.9999, 0.0123, -0.0006, -0.1963;
	H51.row(3) << 0, 0, 0, 1;

	// Motion-lab calibration matrices
	Eigen::Matrix4d X2, X3, T13, T01, T02, H01, H02, HC2;

	X2.row(0) << -0.5042, -0.5417, 0.6726, 0.9685;
	X2.row(1) << -0.8633, 0.3378, -0.3751, -0.2071;
	X2.row(2) << -0.0240, -0.7697, -0.6379, 0.0895;
	X2.row(3) << 0, 0, 0, 1;

	X3.row(0) << 1, 0, 0, 0;
	X3.row(1) << 0, -1, 0, 0;
	X3.row(2) << 0, 0, -1, -0.035;
	X3.row(3) << 0, 0, 0, 1;

	T01.row(0) << -0.0033, 1.0000, 0.0006, -2.9697;
	T01.row(1) << 1.0000, 0.0033, 0.0005, 1.9763;
	T01.row(2) << 0.0005, 0.0006, -1.0000, 2.6628;
	T01.row(3) << 0, 0, 0, 1;

	T02.row(0) << 0.0034, 1.0000, 0.0009, 0.3910;
	T02.row(1) << 1.0000, -0.0034, 0.0017, -1.7810;
	T02.row(2) << 0.0017, 0.0009, -1.0000, 1.6896;
	T02.row(3) << 0, 0, 0, 1;

	T13.row(0) << -0.4972, 0.8676, -0.0050, -1.0820;
	T13.row(1) << 0.8676, 0.4972, 0.0012, 1.5360;
	T13.row(2) << 0.0036, -0.0037, -1.0000, -1.0245;
	T13.row(3) << 0, 0, 0, 1;

	// EQKF
	EQKF KF;
	
	// Create Window and trackbars
	cv::namedWindow("output", 1);

	int iProcessCov = 5;
	int iMeasurementCov = 5;
	cv::createTrackbar("Process Covariance", "output", &iProcessCov, 20);
	cv::createTrackbar("Measurment Covariance", "output", &iMeasurementCov, 20);

	// Print Manual
	PrintManual();

	// Start while llop when camera is ready
	int start = std::clock();
	while (cam.isOpened())
	{
		// Read image from webcam
		cam.read(image);
		image.copyTo(imageCopy);

		// Detect aruco markers
		cv::aruco::detectMarkers(image, dictionary, corners, ids);

		for (unsigned int i = 0; i < 6; i++)
		{
			markerFound[i] = false;
		}

		// Estimate poses of detected aruco markers if founnd
		if (ids.size() > 0)
		{
			// Draw detected markers on image
			cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

			// Estimate pose and position for all detected markers
			cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

			for (unsigned int i = 0; i < ids.size(); i++)
			{
				// Convert and reallocate markers to homogeneous transformation matrices
				switch (ids[i])
				{
				case MARKER_ID_1:
					M1 = VecsToMat(rvecs[i], tvecs[i]);
					a1 = AreaQuadrilateral(corners[i]);

					DrawAxis(imageCopy, M1, camMatrix, distCoeffs, 0.25f * markerLength);
					markerFound[0] = true;
					break;
				case MARKER_ID_2:
					M2 = VecsToMat(rvecs[i], tvecs[i]);
					a2 = AreaQuadrilateral(corners[i]);

					DrawAxis(imageCopy, M2, camMatrix, distCoeffs, 0.25f * markerLength);
					markerFound[1] = true;
					break;
				case MARKER_ID_3:
					M3 = VecsToMat(rvecs[i], tvecs[i]);
					a3 = AreaQuadrilateral(corners[i]);

					DrawAxis(imageCopy, M3, camMatrix, distCoeffs, 0.25f * markerLength);
					markerFound[2] = true;
					break;
				case MARKER_ID_4:
					M4 = VecsToMat(rvecs[i], tvecs[i]);
					a4 = AreaQuadrilateral(corners[i]);

					DrawAxis(imageCopy, M4, camMatrix, distCoeffs, 0.25f * markerLength);
					markerFound[3] = true;
					break;
				case MARKER_ID_5:
					M5 = VecsToMat(rvecs[i], tvecs[i]);
					a5 = AreaQuadrilateral(corners[i]);

					DrawAxis(imageCopy, M5, camMatrix, distCoeffs, 0.25f * markerLength);
					markerFound[4] = true;
					break;
				case MARKER_ID_6:
					M6 = VecsToMat(rvecs[i], tvecs[i]);
					a6 = AreaQuadrilateral(corners[i]);

					DrawAxis(imageCopy, M6, camMatrix, distCoeffs, 0.25f * markerLength);
					markerFound[5] = true;
					break;
				default:
					break;
				}
			}
		}

		// Draw camera coordinate system
		//DrawAxis(imageCopy, Eigen::Matrix4d::Identity(4, 4), camMatrix, distCoeffs, 0.25f * markerLength);

		Eigen::Vector3d pos, posAvg, posEst;
		Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
		Eigen::Vector4d q = Eigen::Vector4d::Zero();
			
		pos << 0, 0, 0;

		int count = 0;
		double aSum = 0.0;
			
		if (markerFound[0])
		{
			T1 = M1;
			pos = pos + a1*T1.block<3, 1>(0, 3);

			q1 = static_cast<Eigen::Matrix3d>(T1.block<3, 3>(0, 0));
			q(0) = q1.w();
			q(1) = q1.x();
			q(2) = q1.y();
			q(3) = q1.z();

			A = a1*(q*q.transpose()) + A;
			aSum = aSum + a1;

			count++;
		}
			
		if (markerFound[1])
		{
			T2 = M2*H21;
			pos = pos + a2*T2.block<3, 1>(0, 3);

			q2 = static_cast<Eigen::Matrix3d>(T2.block<3, 3>(0, 0));
			q(0) = q2.w();
			q(1) = q2.x();
			q(2) = q2.y();
			q(3) = q2.z();

			A = a2*(q*q.transpose()) + A;
			aSum = aSum + a2;

			count++;
		}

		if (markerFound[2])
		{
			T3 = M3*H31;
			pos = pos + a3*T3.block<3, 1>(0, 3);

			q3 = static_cast<Eigen::Matrix3d>(T3.block<3, 3>(0, 0));
			q(0) = q3.w();
			q(1) = q3.x();
			q(2) = q3.y();
			q(3) = q3.z();

			A = a3*(q*q.transpose()) + A;
			aSum = aSum + a3;

			count++;
		}

		if (markerFound[3])
		{
			T4 = M4*H41;
			pos = pos + a4*T4.block<3, 1>(0, 3);

			q4 = static_cast<Eigen::Matrix3d>(T4.block<3, 3>(0, 0));
			q(0) = q4.w();
			q(1) = q4.x();
			q(2) = q4.y();
			q(3) = q4.z();

			A = a4*(q*q.transpose()) + A;
			aSum = aSum + a4;

			count++;
		}
			
		if (markerFound[4])
		{
			T5 = M5*H51;
			pos = pos + a5*T5.block<3, 1>(0, 3);

			q5 = static_cast<Eigen::Matrix3d>(T5.block<3, 3>(0, 0));
			q(0) = q5.w();
			q(1) = q5.x();
			q(2) = q5.y();
			q(3) = q5.z();

			A = a5*(q*q.transpose()) + A;
			aSum = aSum + a5;

			count++;
		}

		if (count > 0)
		{
			// Average the measurements
			posAvg = (1.0 / aSum) * pos;
			A = (1.0 / aSum) * A;

			// Solve eigenvalue problem
			Eigen::EigenSolver<Eigen::Matrix4d> es(A);
				
			Eigen::Vector4d lambda = es.eigenvalues().real();
			Eigen::Matrix4d V = es.eigenvectors().real();

			unsigned int index;
			double maxVal;
			maxVal  = lambda.maxCoeff(&index);


			// Allocate averaged quartenion
			qAvg.w() = V.col(index)(0);
			qAvg.x() = V.col(index)(1);
			qAvg.y() = V.col(index)(2);
			qAvg.z() = V.col(index)(3);
			qAvg.normalize();			

			Eigen::Matrix3d R, R_est;
			R = qAvg.toRotationMatrix();

			H.block<3, 3>(0, 0) = R;
			H.block<3, 1>(0, 3) = posAvg;

			KF.q = static_cast<double>(iProcessCov) / 100.0;
			KF.r = static_cast<double>(iMeasurementCov) / 100.0;
			KF.dt = static_cast<double>(std::clock() - start)/ 1000.0;

			KF.Predict();
			H_est = KF.Correct(H);

				
			qEst = static_cast<Eigen::Matrix3d>(H_est.block<3, 3>(0, 0));

			start = std::clock();
			
			// Draw estimated Arcuo coordinate on top of box
			DrawAxis(imageCopy, H_est, camMatrix, distCoeffs, 0.5f * markerLength);
			//DrawAxis(imageCopy, H_est*X3, camMatrix, distCoeffs, 1.0f * markerLength);
			
			// Reply UDP client with same data amount
			if (udpActive)
			{
				udpSend.dataReady = 1;
				udpSend.posAvg[0] = static_cast<float>(posAvg(0, 0));
				udpSend.posAvg[1] = static_cast<float>(posAvg(1, 0));
				udpSend.posAvg[2] = static_cast<float>(posAvg(2, 0));
				udpSend.posAvg[3] = static_cast<float>(qAvg.w());
				udpSend.posAvg[4] = static_cast<float>(qAvg.x());
				udpSend.posAvg[5] = static_cast<float>(qAvg.y());
				udpSend.posAvg[6] = static_cast<float>(qAvg.z());

				udpSend.posEst[0] = static_cast<float>(H_est(0, 3));
				udpSend.posEst[1] = static_cast<float>(H_est(1, 3));
				udpSend.posEst[2] = static_cast<float>(H_est(2, 3));
				udpSend.posEst[3] = static_cast<float>(qEst.w());
				udpSend.posEst[4] = static_cast<float>(qEst.x());
				udpSend.posEst[5] = static_cast<float>(qEst.y());
				udpSend.posEst[6] = static_cast<float>(qEst.z());
			}
		}
		
		if (udpActive)
		{
			// Robot end-effector
			//DrawAxis(imageCopy, X2.inverse()*ComauFK(udpRead->q), camMatrix, distCoeffs, 1.0f * markerLength);

			// EM-1500 coordinate
			H01 = T01*GetPoseSP(udpRead->posSP1);
			H02 = T02*GetPoseSP(udpRead->posSP2);

			HC2 = X2.inverse()*T13.inverse()*H01.inverse()*H02;
			//DrawAxis(imageCopy, HC2, camMatrix, distCoeffs, 1.0f * markerLength);
		}

		// Show video with detected markers
		cv::imshow("output", imageCopy);

		// Keyboard events
		switch (cv::waitKey(1))
		{
		case 'c' :
			// Calibrate Aruco side offsets
			if (markerFound[0] && markerFound[1])
			{
				H21 = M2.inverse()*M1;
				WriteMatToCSV(H21, fileH21);
				cubeCalibCount[0]++;
			}

			if (markerFound[0] && markerFound[2])
			{
				H31 = M3.inverse()*M1;
				WriteMatToCSV(H31, fileH31);
				cubeCalibCount[1]++;
			}

			if (markerFound[0] && markerFound[3])
			{
				H41 = M4.inverse()*M1;
				WriteMatToCSV(H41, fileH41);
				cubeCalibCount[2]++;
			}

			if (markerFound[0] && markerFound[4])
			{
				H51 = M5.inverse()*M1;
				WriteMatToCSV(H51, fileH51);
				cubeCalibCount[3]++;
			}
			std::cout << "H21 count = " << cubeCalibCount[0] << std::endl;
			std::cout << "H31 count = " << cubeCalibCount[1] << std::endl;
			std::cout << "H41 count = " << cubeCalibCount[2] << std::endl;
			std::cout << "H51 count = " << cubeCalibCount[3] << std::endl << std::endl;

			break;
		case VK_SPACE :
			// Save snapshot to .jpg file
			saveName = "./cam_snapshots/image" + std::to_string(saveImageCount) + ".jpg";
			std::cout << "image" << saveImageCount << ".jpg was saved." << std::endl;
			cv::imwrite(saveName, imageCopy);
			saveImageCount++;
			break;
		case VK_RETURN :
			// Write data to measure file
			fileCAM << H_est(0, 3) << "," << H_est(1, 3) << "," << H_est(2, 3) << ",";
			fileCAM << qEst.w() << "," << qEst.x() << "," << qEst.y() << "," << qEst.z() << "\n";

			fileROB << udpRead->q[0] << "," << udpRead->q[1] << "," << udpRead->q[2] << ",";
			fileROB << udpRead->q[3] << "," << udpRead->q[4] << "," << udpRead->q[5] << "\n";

			std::cout << "Hand-Eye Save Count = " << handEyeCount << std::endl;
			handEyeCount++;
			break;
		case VK_ESCAPE :
			// Stop UDP thread and release cam
			fileCAM.close();
			fileROB.close();
			fileH21.close();
			fileH31.close();
			fileH41.close();
			fileH51.close();
			// Stop camera capture
			cam.release();
			// CLose udpThread
			closeUdp = true;
			udpThread.join();
			Sleep(1000);
			break;
		}
	}

	// Close all OpenCV windows
	cv::destroyAllWindows();

	return 0;
};