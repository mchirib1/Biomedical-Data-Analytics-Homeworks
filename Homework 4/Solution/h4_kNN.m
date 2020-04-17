% BENG420/520
% Homework #3 - apply k-NN on cancer gene expression classification
% Data is from [Phan et al., Nature, 2001]
% qwei
% 4/1/2020

clear all
close all

load h4_data.mat


%Where D is the number of features and M/N are the number of examples
traind = [(geneexpTrain) ,encode(tumortypeTrain)];  %randomize samples
rand_train = traind(randperm(size(traind,1)),:);    


train_x =rand_train(:,1:10);  %NxD Matrix
train_y = rand_train(:,11);   %Nx1 Vector

test_x = geneexpTest;          %MxD Vector
test_y = encode(tumortypeTest) %Mx1 Vector

%Iterate number of nearest neighbors
for l = 1:20
    k = l;
    
    % calculating the euclidean distance of the test samples from training
    for i = 1:length(train_x)
        for j = 1: length(test_x)
            %{
        Compute the distances of the test values to the training values.
        The Column is the euclidian distance of ONE specific Test sample
        sample to the each of the given training samples.
            %}
            d(i,j) = sqrt(sum(test_x(j,:)-train_x(i,:)).^2);
        end
    end
    
    %sort ascending to get the nearest neighbors indicies
    [c,idx] = sort(d);
    
    %stores the labels of each of the neighbors in distance order
    dist_to_lab = train_y(idx);
    
    %chooses a guess for the class of each of the 20 samples based on most
    %   most common class in the nearest k neighbors
    for j = 1:length(test_x)
        m = mode(dist_to_lab(1:k,j));
        classe(j) = m;
    end

    %calculates whether or not the predicted class matches with the true
    %class
    for i = 1:length(classe)
        if classe(:,i) == test_y(i,:)
            acc(:,i) = 1;
        else
            acc(:,i) = 0 ;
        end
    end
   %calculates the accuracy 
    accuracy(l,:) = [(sum(acc)/length(acc))*100,l];

end
%plots the accuracy as a function of k
figure(1)
plot(accuracy(:,2),accuracy(:,1))



%Functions
%==========================================================================
%       EW = [1 0 0 0]';
%       BL  = [0 1 0 0]';
%       NB  = [0 0 1 0]';
%       RM  = [0 0 0 1]';

function cancers = decode(diagnosis)
for i = 1:length(diagnosis)
    if diagnosis(i,:) == [1 0 0 0]
        cancers(i,:) = ('EW');
    end
   
    if diagnosis(i,:) == [0 1 0 0]
        cancers(i,:) = ('BL');
    end
    
    if diagnosis(i,:) == [0 0 1 0]
        cancers(i,:) = ('NB');
    end
    
    if diagnosis(i,:) == [0 0 0 1]
        cancers(i,:) = ('RM');
    end

end


end

function classes = encode(Labels)
for i = 1:length(Labels)
    if Labels(i,1) == "EW"
        classes(i,1) = 1;
    end
    
    if Labels(i,1) == "BL"
        classes(i,1) = 2;
        
    end
    
    if Labels(i,1) == "NB"
        classes(i,1) = 3;
        
    end
    
    if Labels(i,1) == "RM"
        classes(i,1) = 4;
        
    end
end
classes
end