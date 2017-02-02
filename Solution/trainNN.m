function nets = trainNN(data)
% Nikhil Kamthe
% 861245635
% 12/06/2016
% CS 229
% Final Project
%
% This function trains the Neural Network and stores the classifier on the
% file system.

% Initialize parameters
classifierCount = 15;

% The following part creates a bag of classifiers (Neural Networks) using bootstrap data.
% The average of these classifiers is then used for final prediction.
[nets] = bagging(data,classifierCount);

% Storing the built classifiers on file system.
save('nets','nets');

end