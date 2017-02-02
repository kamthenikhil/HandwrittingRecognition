function test3()
% error rate vs lambda (regularization)

data = load('handwriting.data','-ascii');


data = data(10,:);

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
k = 10;
sampleSize = 100;

lambdas = linspace(0,1,sampleSize);

trainErrors = zeros(1,sampleSize);
testErrors = zeros(1,sampleSize);

for i = 1:sampleSize
    net = patternnet(k);
    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.testRatio = 20/100;
    net.trainFcn = 'trainscg';

    net.performParam.regularization = 0.1;
    [net,tr] = train(net,inputs,targets);
    
    save net;
    save tr;
    
    clearvars net tr;
    
    load net;
    load tr;
    
    trainIndices = tr.trainInd;
    trainOutputs = net(inputs(:,trainIndices));
    trainOutputs = process(trainOutputs);
    trainErrors(i) = fetchErrorRate(targets,trainOutputs,trainIndices);

    testIndices = tr.testInd;
    testOutputs = net(inputs(:,testIndices));
    testOutputs = process(testOutputs);
    testErrors(i) = fetchErrorRate(targets,testOutputs,testIndices);
end

plot(lambdas,trainErrors,'r');
hold on;
plot(lambdas,testErrors,'b');
xlabel('lambdas');
ylabel('error rate');
title('Error Rate vs Lambda');
legend('Train Data','Testing Data');
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