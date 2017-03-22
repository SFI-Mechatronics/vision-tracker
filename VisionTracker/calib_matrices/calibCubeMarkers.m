close all
clear variables
clc

% Read calibration data
folder = '.\calib0\';
dataH21 = dlmread([folder,'dataH21.txt'],',');
dataH31 = dlmread([folder,'dataH31.txt'],',');
dataH41 = dlmread([folder,'dataH41.txt'],',');
dataH51 = dlmread([folder,'dataH51.txt'],',');

% Average data for H21
[H21,H21Avg] = AverageData(dataH21);

% Average data for H31
[H31,H31Avg] = AverageData(dataH31);

% Average data for H41
[H41,H41Avg] = AverageData(dataH41);

% Average data for H51
[H51,H51Avg] = AverageData(dataH51);

H21Avg
H31Avg
H41Avg
H51Avg
