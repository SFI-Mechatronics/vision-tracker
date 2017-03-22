close all
clear variables
clc

% Load calib matrices
load('motionLabCalib.mat')

% Load data from measurements
folder = '.\calib1\';
dataCAM = dlmread([folder,'dataCAM.txt'],',');
dataROB = dlmread([folder,'dataROB.txt'],',');


N = length(dataCAM(:,1));
C = zeros(4,4,N);
T = zeros(4,4,N);

for i = 1:N
    t = dataCAM(i,1:3)';
    R = quat2rotm(dataCAM(i,4:7));
    C(:,:,i) = eye(4,4);
    C(1:3,4,i) = t;
    C(1:3,1:3,i) = R;
    
    q = dataROB(i,:)';
    cart = ComauFK(q/180*pi,'comau','none','false');
    
    T(:,:,i) = cart.T06;
end

A = zeros(4,4,N-1);
B = zeros(4,4,N-1);
for i = 2:N
    A(:,:,i-1) = InvH(T(:,:,i))*T(:,:,i-1);
    B(:,:,i-1) = InvH(C(:,:,i))*C(:,:,i-1);
end

X1 = HandEyeParkMartin(A,B)
%%
for i = 1:N
    X(:,:,i) = T(:,:,i)*X1*InvH(C(:,:,i));
end

q = rotm2quat(X(1:3,1:3,:));
qAvg = QuatAvgMarkley(q);

X2 = eye(4,4);
X2(1:3,1:3) = quat2rotm(qAvg);
X2(1,4) = mean(X(1,4,:));
X2(2,4) = mean(X(2,4,:));
X2(3,4) = mean(X(3,4,:))

% Save hand-eye calibration matrices
save('C:\Users\Sondre\Dropbox\MATLAB Custom Functions\handEyeCalib','X1','X2')


%% Animation figure settings
h = figure;
h.Position = [380 200 1000 800];
axis equal, grid on
view(-25,35)
xlabel('X')
ylabel('Y')
zlabel('Z')
xlim([-0.2,3]);
ylim([-2,0.5]);
zlim([-2,0.5]);


for i = 1:N

    E = T(:,:,i) - X2*C(:,:,i)*InvH(X1);
    if ishandle(h)
        % Transform objets
        hold on    
            cla

            PlotAxes(T(:,:,i)*X1,0.3,2)
            PlotAxes(X2*C(:,:,i),0.5,1)

            PlotAxes(eye(4,4),0.2,2)
        hold off


        drawnow;
        pause(0.5);
    else
        break;
    end
    
end

e = sum(sum(abs(E)))



