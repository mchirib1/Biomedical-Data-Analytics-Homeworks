% BENG420/520 Homework #3
% Compare a neuron with a threshold transfer function to a neuron with
% logistic sigmoid transfer function
%
% qwei

close all
clear all

% generate 2D feature values for two classes of data points
X1=[rand(1,100);rand(1,100);ones(1,100)];   % class '+1'
X2=[rand(1,100);1+rand(1,100);ones(1,100)]; % class '-1'
X=[X1,X2]';

% define class labels as being -1 or +1
Y=[-ones(1,100),ones(1,100)]';

% randomly initialize the weight vector
theta0 = rand(1, 3);

% plot the data points
figure(1); hold on
plot(X1(1,:),X1(2,:),'b.');
plot(X2(1,:),X2(2,:),'r.');
title('perceptron');

% use a learning rate of 1
learningRate = 1;

% max number of iterations
niter = 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOU: call perceptron neuron learning to learn the weights
thetaP = perceptron_neuron(X,Y,theta0,niter,learningRate)

% YOU: use the learned weights to predict labels of all the training data

%Calculate the neuron activation for each training example
activation = X*(thetaP');

predicted_label = zeros(size(activation));
%If the activation breaks threshold 0, predict label 1
for i = 1:size(X,1);
  if activation(i,:)>=0;
    predicted_label(i,:) = 1;
    
  %If activation does not break threshold of 1, predict label -1
  else 
    predicted_label(i,:) = -1;
  end
end

% YOU: print the training accuracy

accuracy = zeros(size(activation));
%Iterate through the predicted labels and add to accuracy if error
for i = 1:size(X,1);
  if  predicted_label(i,:)~=Y(i,:);
    accuracy(i,:) = 1; 
  end
end
%Compute the error and divide the error by the number of examples to get train accuracy
error_perceptron = sum(accuracy)
train_accuracy_perceptron = ((1-(error_perceptron/size(X,1)))*100)

% YOU: plot predicted labels as circles of the same color over original data

%selects the instances of the two classes.
class1 = predicted_label(:) ~= -1;
class2 = predicted_label(:) ~= 1;

%organizes the data points based on the class instance
predclass1data = [X(class1,1),X(class1,2)]'; %predicting the -1 class
predclass2data = [X(class2,1),X(class2,2)]'; %predicting the +1 class

%plot the predicted class on the figure 1
plot(predclass1data(1,:),predclass1data(2,:),'ro','markersize',10);
plot(predclass2data(1,:),predclass2data(2,:),'bo','markersize',10);

legend('class -1','class +1','pred +1','pred -1')

% plot decision boundary
line([0 1],[-thetaP(3)/thetaP(2) -(thetaP(3)+thetaP(1))/thetaP(2)]);
xlabel('x1');
ylabel('x2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOU: call logistic neuron learning to learn the weights
thetaL = logistic_neuron(X,Y,theta0,niter,learningRate)

% YOU: use the learned weights to predict labels of all the training data

%Calculate the neuron activation for each training example
activation = X*(thetaL');

predicted_label = zeros(size(activation));
%If the activation breaks threshold 0, predict label 1
for i = 1:size(X,1);
  if activation(i,:)>= (0);
    predicted_label(i,:) = 1;
    
  %If activation does not break threshold of 1, predict label -1
  else 
    predicted_label(i,:) = -1;
  end
end

% YOU: print the training accuracy

accuracy = zeros(size(activation));
%Iterate through the predicted labels and add to accuracy if error
for i = 1:size(X,1);
  if  predicted_label(i,:)~=Y(i,:);
    accuracy(i,:) = 1; 
  end
end
%Compute the error and divide the error by the number of examples to get train accuracy
error_logistic = sum(accuracy)
train_accuracy_logistic = ((1-(error_logistic/size(X,1)))*100)

% YOU: plot predicted labels as circles of the same color over origianl data
%selects the instances of the two classes.
class1 = predicted_label(:) ~= -1;
class2 = predicted_label(:) ~= 1;

%organizes the data points based on the class instance
predclass1data = [X(class1,1),X(class1,2)]'; %predicting the -1 class
predclass2data = [X(class2,1),X(class2,2)]'; %predicting the +1 class

%Make a new figure for the Logistic Neuron
figure(2); hold on
plot(X1(1,:),X1(2,:),'b.');
plot(X2(1,:),X2(2,:),'r.');
title('log perceptron');

%plot the logistic predictions on figure 2
plot(predclass1data(1,:),predclass1data(2,:),'rs','markersize',10);
plot(predclass2data(1,:),predclass2data(2,:),'bs','markersize',10);

legend('class -1','class +1','pred +1','pred -1')

% plot decision boundary
line([0 1],[-thetaL(3)/thetaL(2) -(thetaL(3)+thetaL(1))/thetaL(2)]);
xlabel('x1');
ylabel('x2');


%{ 
Discussion Section:
    It seems like generally the accuracy of the logistic neuron is 
    more often higher than the perceptron. Second of all the slopes are
    not always the same.  Sometimes the perceptron decision boundry has a
    negative slope and the logistic neuron has a positive slope.  This
    occurs when each of the algorithms are based on the same initial theta
    values. 
    
    The differences in the algorithms is that the logistic neuron has a
    nonlinear activation funciton.  The closer closer the activation value
    is to zero the less "sure" the neuron is that the specific data point
    is correctly classified. The theta update then calculates the error and
    scales the weights so activations are closer to either 0 or 1. This 
    is opposed to the unit step nature of the original threshold function,
    where once theta is updated by a unit step each time until a certain
    threshold is reached.
%}