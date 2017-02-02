function test4()
% error rate vs number of classifiers (bagging)

data = load('handwriting.data','-ascii');

classifierCount = 40;
[m,d] = size(data);
testIndices = randsample(m,round(m/5));
trainIndices = setdiff(1:m,testIndices);

[nets] = bagging(data(trainIndices,:),classifierCount);

data = data(testIndices,:);
[m,d] = size(data);
y = data(:,1);
x = data(:,2:d);
targets = zeros(m,26);
for i = 1:m
    index = y(i,1);
    targets(i,index+1) = 1;
end
testInputs = x';
testTargets = targets';
testErrors = zeros(1,classifierCount);
classifierCounts = 1:classifierCount;

for count = classifierCounts
    testOutputs = zeros(26,m);
    for i = 1:count
        net = nets{i};
        testOutputs = testOutputs + net(testInputs);
    end
    testOutputs = process(testOutputs./count);
    testErrors(count) = fetchErrorRate(testTargets,testOutputs,1:m);
end
plot(classifierCounts,testErrors,'b');
xlabel('number of classifiers');
ylabel('error rate');
title('Bagging');
legend('Testing Data');
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