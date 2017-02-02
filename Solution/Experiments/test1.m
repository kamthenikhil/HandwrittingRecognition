function test1()
% error rate vs hidden units

data = load('handwriting.data','-ascii');

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
net = patternnet(150);
net.divideParam.trainRatio = 80/100;
net.divideParam.testRatio = 20/100;
net.trainFcn = 'trainscg';
net.layers{1}.transferFcn = 'tansig';
net.performParam.regularization = 0.1;
[net,tr] = train(net,inputs,targets);

trainIndices = tr.trainInd;
trainOutputs = net(inputs(:,trainIndices));
trainOutputs = process(trainOutputs);
trainError = fetchErrorRate(targets,trainOutputs,trainIndices);

testIndices = tr.testInd;
testOutputs = net(inputs(:,testIndices));
testOutputs = process(testOutputs);
testErrors = fetchErrorRate(targets,testOutputs,testIndices)
end

function errorRate = fetchErrorRate(targets,outputs,indices)
mismatches = 0;
counter = 1;
for i = indices
    if isequal(targets(:,i),outputs(:,counter))==0
        mismatches = mismatches + 1;
    end
    counter = counter + 1;
end
errorRate = mismatches/length(indices)*100;
end

function output = process(output)
[m,d] = size(output);
for i = 1:d
   [maximum,index] = max(output(:,i));
   output(:,i) = zeros(26,1);
   output(index,i) = 1;
end
end