function main()
% Nikhil Kamthe
% 861245635
% 12/06/2016
% CS 229
% Final Project

data = load('handwriting.data','-ascii');
[m,d] = size(data);
testIndices = randsample(m,round(m/5));
trainIndices = setdiff(1:m,testIndices);
x = data(testIndices,2:d);
y = data(testIndices,1);

trainNN(data(trainIndices));
accuracy = testNN(x,y);

disp(accuracy);
end