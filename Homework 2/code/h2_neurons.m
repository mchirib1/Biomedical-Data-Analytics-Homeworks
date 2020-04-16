% BENG420/520 Homework #2
% learning a perceptron
%
% qwei

close all
clear all

% generate 2D feature vablues for two classes of data points
X1=[rand(1,100);rand(1,100);ones(1,100)];   % class '+1'
X2=[rand(1,100);1+rand(1,100);ones(1,100)]; % class '-1'
X=[X1,X2]';

% define class labels as being -1 or +1
Y=[-ones(1,100),ones(1,100)]';

% randomly initialize the weigth vector
theta0 = rand(1, 3);

% plot the data points
figure; hold on
plot(X1(1,:),X1(2,:),'b.');
plot(X2(1,:),X2(2,:),'r.');
title('perceptron');

% use the same learning rate for both neurons
learningRate = 1;

% max number of iterations
niter = 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOU: call perceptron neuron learning to learn the weights
thetaP = perceptron_neuron(X,Y,theta0,niter,learningRate);

% YOU: use the learned weights to predict labels of all the training data

% YOU: print the training accuracy

% YOU: plot predicted labels as circles of the same color over origianl data

legend('class -1','class +1','pred -1','pred +1')
% plot decision boundary
plotpc(thetaP(1:2),thetaP(3));
xlabel('x1');
ylabel('x2');

figure;hold on
plot(X1(1,:),X1(2,:),'b.')
plot(X2(1,:),X2(2,:),'r.')
title('logistic neuron');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOU: call logistic neuron learning to learn the weights
thetaL = logistic_neuron(X,Y,theta0,niter,learningRate);

% YOU: use the learned weights to predict labels of all the training data

% YOU: print the training accuracy

% YOU: plot predicted labels as circles of the same color over origianl data

legend('class -1','class +1','pred -1','pred +1')
% plot decision boundary
plotpc(thetaL(1:2),thetaL(3));
xlabel('x1');
ylabel('x2');