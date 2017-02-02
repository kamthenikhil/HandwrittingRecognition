function accuracy = testNN(x,y)
% Nikhil Kamthe
% 861245635
% 12/06/2016
% CS 229
% Final Project
%
% This method is used to use the stored Neural Networks to predict the
% outputs for input data.

load('nets');
inputs = x';
[m,d] = size(x');
classifierCount = length(nets);

targets = zeros(26,d);
for i = 1:classifierCount
    net = nets{i};
    targets = targets + net(inputs);
end
y_pred = process(targets./classifierCount);
errorRate = 100*size(y_pred(y_pred~=y))/size(y_pred);
accuracy = 100-errorRate;
end

function y = process(targets)
% This method converts targets of Neural Network to y.

[m,d] = size(targets);
outputs = zeros(1,d);
for i = 1:d
   [maximum,index] = max(targets(:,i));
   outputs(1,i) = index-1;
end
y = outputs';
end