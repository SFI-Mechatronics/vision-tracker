function [H,HAvg] = AverageData(data)


for i = 1:length(data(:,1))
    H(:,:,i) = reshape(data(i,:),[4,4]);    
end

q = rotm2quat(H(1:3,1:3,:));
qAvg = QuatAvgMarkley(q);
pAvg = mean(H(1:3,4,:),3);

HAvg = eye(4,4);
HAvg(1:3,1:3) = quat2rotm(qAvg);
HAvg(1:3,4) = pAvg;



