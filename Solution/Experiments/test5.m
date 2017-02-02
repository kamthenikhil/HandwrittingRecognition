function test5()

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

trainErrors = zeros(1,2);
testErrors = zeros(1,2);

net = patternnet(k);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.testRatio = 20/100;
net.trainFcn = 'trainscg';

net.performParam.regularization = 0.1;
[net,tr] = train(net,inputs,targets);

trainIndices = tr.trainInd;
trainOutputs = net(inputs(:,trainIndices));
trainOutputs = process(trainOutputs);
trainErrors(1) = fetchErrorRate(targets,trainOutputs,trainIndices);

testIndices = tr.testInd;
testOutputs = net(inputs(:,testIndices));
testOutputs = process(testOutputs);
testErrors(1) = fetchErrorRate(targets,testOutputs,testIndices);

inputs = inputs([66   73   60   79   69   64   67  125   59   54   18    1   78   45    7   50   36   80   55   49   27   71   35   48   96  110  118   52   74  112   94   28  104  122   14   51   91   86   68   85   61   57   43   22   20   65   72   90   46    5   95   63  116  105   39  117   13    9  103  126   15   70   76   77   38  127   23   17  129   98   44   42  120   10  102  108   87   29   25  121  111   11   26   89   84   37   34  123   83   92    3   21   58   56   99    8    2    6   24  100   47   93   31   33   81   12  115    4  113  114   88   30   40   53  109],:);
net = patternnet(k);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.testRatio = 20/100;
net.trainFcn = 'trainscg';

net.performParam.regularization = 0.1;
[net,tr] = train(net,inputs,targets);

trainIndices = tr.trainInd;
trainOutputs = net(inputs(:,trainIndices));
trainOutputs = process(trainOutputs);
trainErrors(2) = fetchErrorRate(targets,trainOutputs,trainIndices);

testIndices = tr.testInd;
testOutputs = net(inputs(:,testIndices));
testOutputs = process(testOutputs);
testErrors(2) = fetchErrorRate(targets,testOutputs,testIndices);

trainErrors
testErrors

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