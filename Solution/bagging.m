function [nets] = bagging(data,classifierCount)
% Nikhil Kamthe
% 861245635
% 12/06/2016
% CS 229
% Final Project
%
% This method generates a pool of Neural Networks which are then used for
% bagging Neural Network classifier. It first generates a bootstrap dataset
% from the input dataset and uses it to build each of these classifiers.

nets = cell(1,classifierCount);
for i = 1:classifierCount
    bootstrapData = bootstrap(data);
    nets{i} = trainNeuralNetwork(bootstrapData);
end

end

function bootstrapData = bootstrap(data)
% This method generates a bootstrap ddataset using the input
% dataset. It picks random indices from input data (without replacement)
% and uses those indices to create the bootstrap dataset of same size.

[m,d] = size(data);
indices = randi([1 m],m,1);
bootstrapData = data(indices,:);
end

function net = trainNeuralNetwork(data)
% This method trains the Neural Netwrok from the given data. The parameters
% used for building the classifiers are selected after running a number of
% experiments.

[m,d] = size(data);
y = data(:,1);
x = data(:,2:d);
targets = zeros(m,26);
for i = 1:m
    index = y(i,1);
    targets(i,index+1) = 1;
end
inputs = x';
targets = targets';
k = 150;
net = patternnet(k);
net.divideParam.trainRatio = 100/100;
net.trainFcn = 'trainscg';
net.layers{1}.transferFcn = 'tansig';
net.performParam.regularization = 0.1;
[net] = train(net,inputs,targets);
end