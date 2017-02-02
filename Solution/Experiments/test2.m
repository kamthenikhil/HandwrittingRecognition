function test2()
% error rate vs training set size
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
k = 150;
testDatasetSize = round(m/5);
testDatasetStartIndex = m-testDatasetSize+1;
testDatasetIndices = testDatasetStartIndex:m;
trainDatasetSizes = linspace(testDatasetSize,m-testDatasetSize,20);
counter = 1;
count = length(trainDatasetSizes);
trainErrors = zeros(1,count);
testErrors = zeros(1,count);
for trainDatasetSize = trainDatasetSizes
    net = patternnet(k);
    net.divideFcn = 'divideind'; 
    net.divideParam.trainInd = 1:trainDatasetSize;
    net.divideParam.testInd = testDatasetIndices;
    
    net.trainFcn = 'trainscg';
    net.performParam.regularization = 0.1;
    [net,tr] = train(net,inputs,targets);

    trainIndices = tr.trainInd;
    trainOutputs = net(inputs(:,trainIndices));
    trainOutputs = process(trainOutputs);
    trainErrors(counter) = fetchErrorRate(targets,trainOutputs,trainIndices);

    testIndices = tr.testInd;
    testOutputs = net(inputs(:,testIndices));
    testOutputs = process(testOutputs);
    testErrors(counter) = fetchErrorRate(targets,testOutputs,testIndices);
    counter = counter + 1;
end
trainDatasetSizes
trainErrors
testErrors
plot(trainDatasetSizes,trainErrors,'r--');
hold;
plot(trainDatasetSizes,testErrors,'b-');
title('Error Rate vs Training Set Size');
xlabel('training set size');
ylabel('error rate');
legend('training data','testing data');
hold off;
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