% BENG420/520
% Homework #4 - apply ANN on cancer gene expression classification
% Data is from [Phan et al., Nature, 2001]
% qwei
% 4/1/2020

clear all
close all

% load data
load h4_data.mat

% YOU: convert class labels properly to Matlab format

%Encode label vectors
train_y = encode(tumortypeTrain);
test_y = encode(tumortypeTest);

%Input matrices
train_x = geneexpTrain;
test_x = geneexpTest;

% YOU: build your single layer LINEAR ANN that takes 10 inputs and gives 4
% outputs; specify parameters on training the classifier as needed

%one input and one layer but only connect the bias, inputs, and outputs
net = network(1,1,true,true,false,true);

%{
using example inputs and outputs I note that the number of weighted
elements is 44 which corresponds with the Nature Paper.  This automatically
adjusts the parameters.
%}
net.inputs{1}.exampleInput = train_x';
net.outputs{1}.exampleOutput = train_y';

net.IW{1}; %shoudl be 10x4 matrix 
net.LW{1}; %should be empty
net.b{1};  %should be a vector of 4 values

net.performFcn = 'mse'; %"sum smeasures the error (1/2sum((t-y)^2)
net.performParam.normalization = 'standard'; %normalizes the distance to unity as done in the paper
%{
for some reason when I use the logsig function I get convergence to a much
better cost function minima then if I use either tansig or hardlim
%}
net.layers{1}.transferFcn = 'logsig'; %tansig; hardlim




%{
train using the Levenberg-Marquardt backpropagation algorithm. this
algorithm seems to be a type of gauss-newton fixed point iteration. which
I know Newtons method will only converge to global minimum with sufficently 
close initial guess.
%}
net.trainFcn = 'trainlm'; 
net.divideFcn = 'dividetrain'; %assigns all target labels to the training set 

%initialize the network
% net.initFcn = 'initlay';
% net = init(net); 

%{
I find it interesting there is no initalization in the input weights or
bias. I guess initialization only occurs with the presence of LW's.  Might
be due to the fact only layers have a  Nguyen-Widrow initialization
function option?
%}
net.IW{1};
net.LW{1};
net.b{1};

%just used for comparison to premade FF net
net2 = feedforwardnet([]);
net2.performParam.normalization = 'standard';
[trainedNet2,tr_ff] = train(net2,train_x',train_y');

% YOU: train the ANN classifier using training data
[net,tr] = train(net,train_x',train_y');

%notice the change in the weights and bias. the IW and b are the same each time
net.IW{1};
net.LW{1};
net.b{1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Examine the performance of your ANN classifier on the 63 training
% examples and report training accuracy

[t] = ((net(train_x')).*100)';
[t_ff] = ((trainedNet2(train_x')).*100)';

%{
here you can see that the feedfoward and my custom network reach at least
local cost function minima after 1 epoch. there is the same amount of error
every time.

This is interesting because even though the same cost function minima is
found the predictions change due to the random shuffling of the validation
and test sets built into the feedforword network:

         seen when viewing "t" and "t_ff" in variable explorer

Below I will compare the labeling accuracies of each of the networks.
%}
figure(5);
plotperform(tr);
figure(6);
plotperform(tr_ff);



% YOU: predict the class labels of the training examples using your trained
% ANN classifier
%====================================================
%My network
%{
Here the class labels are predicted using either the max value or using the
95% cutoff in the paper
%}
[c,p] = prediction(t);


%decode to diagnose
Consensus_diagnosis = decode(c);
Percentile_diagnosis = decode(p);

Consensus_accuracy = labeling_accuracy(c,train_y);
Percentile_accuracy = labeling_accuracy(p,train_y);
%====================================================
%feedfoward network
[cff,pff] = prediction(t_ff);

%decode to diagnose
Consensus_diagnosisff = decode(cff);
Percentile_diagnosisff = decode(pff);

Consensus_accuracyff = labeling_accuracy(cff,train_y);
Percentile_accuracyff = labeling_accuracy(pff,train_y);

Total_train_accuracies = [Consensus_accuracy,Percentile_accuracy;...
                    Consensus_accuracyff,Percentile_accuracyff]
                    
%====================================================
% YOU: plot confusion matrix
%Confusion matrices are commented out becuase they were annoying popping up

confmat_c = confusionmat(tumortypeTrain,Consensus_diagnosis);
figure(1)
confusionchart(confmat_c);
% confmat_p = confusionmat(tumortypeTrain,Percentile_diagnosis);
% figure(2)
% confusionchart(confmat_p);

confmat_cff = confusionmat(tumortypeTrain,Consensus_diagnosisff);
figure(2)
confusionchart(confmat_cff);
% confmat_pff = confusionmat(tumortypeTrain,Percentile_diagnosisff);
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Examine the performance of your ANN classifier on the test
% examples and report test accuracy

t_t = ((net(test_x')).*100)';
t_tff = ((trainedNet2(test_x')).*100)';% YOU: predict the class labels of the test examples using your trained
%====================================================
%My network
% ANN classifier
[ct,pt] = prediction(t_t);

%decode to diagnose
Consensus_diagnosist = decode(ct);
Percentile_diagnosist = decode(pt);

Consensus_accuracyt = labeling_accuracy(ct,test_y);
Percentile_accuracyt = labeling_accuracy(pt,test_y);
%====================================================
%feedfoward network
[cfft,pfft] = prediction(t_tff);

%decode to diagnose
Consensus_diagnosisfft = decode(cfft);
Percentile_diagnosisfft = decode(pfft);

Consensus_accuracyfft = labeling_accuracy(cfft,test_y);
Percentile_accuracyfft = labeling_accuracy(pfft,test_y);

Total_test_accuracies = [Consensus_accuracyt,Percentile_accuracyt;...
                    Consensus_accuracyfft,Percentile_accuracyfft]

% YOU: plot confusion matrix
confmat_ct = confusionmat(tumortypeTest,Consensus_diagnosist);
figure(3)
confusionchart(confmat_ct);

confmat_cfft = confusionmat(tumortypeTest,Consensus_diagnosisfft);
figure(4)
confusionchart(confmat_cfft);
%Functions
%==========================================================================
%       EW = [1 0 0 0]';
%       BL  = [0 1 0 0]';
%       NB  = [0 0 1 0]';
%       RM  = [0 0 0 1]';
function accuracy = labeling_accuracy(p,y)
for i = 1:length(p)
    if p(i,:) == y(i,:)
        a(i) = 1;
    else
        a(i) = 0;
        
    end
end
accuracy = (sum(a)/length(a))*100;
end

function [consensus,percentile] = prediction(A)
%use consensus
for j = 1:size(A,2)
    for i = 1:length(A)
        %use the consensus to predict class
        if A(i,j) == max(A(i,:))
            consensus(i,j) = 1;
        else
            consensus(i,j)=0;
        end
    end
end

%use Percentile
for j = 1:size(A,2)
    for i = 1:length(A)
        %use the consensus to predict class
        if A(i,j) > 95
            percentile(i,j) = 1;
        else
            percentile(i,j)=0;
        end
    end
end
percentile;
consensus;
end

function cancers = decode(diagnosis)
for i = 1:length(diagnosis)
    if diagnosis(i,:) == [1 0 0 0]
        cancers(i,:) = cellstr('EW');
    end
   
    if diagnosis(i,:) == [0 1 0 0]
        cancers(i,:) = cellstr('BL');
    end
    
    if diagnosis(i,:) == [0 0 1 0]
        cancers(i,:) = cellstr('NB');
    end
    
    if diagnosis(i,:) == [0 0 0 1]
        cancers(i,:) = cellstr('RM');
    end
    
end


end

function classes = encode(Labels)
classes = zeros(4,length(Labels));
for i = 1:length(Labels)
    if Labels(i,1) == "EW"
        classes(1,i) = 1;
    end
    
    if Labels(i,1) == "BL"
        classes(2,i) = 1;
        
    end
    
    if Labels(i,1) == "NB"
        classes(3,i) = 1;
        
    end
    
    if Labels(i,1) == "RM"
        classes(4,i) = 1;
        
    end
end
classes = classes';
end