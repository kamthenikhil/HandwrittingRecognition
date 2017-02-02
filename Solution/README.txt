- Nikhil Kamthe
- 861245635
- 12/06/2016
- CS 229

Instructions to run:

1. Running on Test dataset: In order to run the previously built classifier on new data, use the funtion testNN. This function takes two arguments viz. x (input data with samples as rows) and y (a column vector of output data). This function uses previously built classifier to predict the output for this new data and returns the accuracy after comparing it with the supplied output.
e.g.
data = load('test.data','-ascii');
[m,d] = size(data);
x = data(:,2:d);
y = data(:,1);
accuracy = testNN(x,y);


2. Retrain the classifier: In order to retrain the classifier, use the trainNN function which takes data as input (each row in data is a sample such that the first element is the output and remaining elements correspond to the features).
e.g.
data = load('train.data','-ascii');
trainNN(data);

NOTE: Refer main.m file, which contains sample code for executing both of these cases.