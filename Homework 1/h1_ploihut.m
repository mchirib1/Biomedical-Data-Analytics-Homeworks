 %Reads in the Data and the class labeles and aligns them with the Data pts.
data = [features(:,1),features(:,2),classlabels];

%selects the instances of the two classes.
class1 = classlabels(:) ~= 2;
class2 = classlabels(:) ~= 1;

%8000x2 matrix that contains the data points for class 1 and 2 respective.
%The matrices are basically concatinations of x value array and y value array.
class1data = [data(class1,1),data(class1,2)];
class2data = [data(class2,1),data(class2,2)];

%Plotting the Data pts for the two classes.
%The plot looks like some sort of sinusoidal curve or wave.
figure(1)
plot(data(class1,1),data(class1,2),'ro',data(class2,1),data(class2,2),'bs')
%===============================================================================
%find the shape of the train and test data
n_train = [features(1:500,:)];
sizen_train = size(n_train);

n_test = [features(4001:4500,:)];
sizen_test = size(n_test,1)

#plotting the training data just to help visualize it
figure(2)
plot(class1data(1:500,:),'ro',class2data(1:500,:),'bs')
%===============================================================================
#Here are the parameters for the knn function
k = 5;
data = features(1:500,:);
labels = classlabels(1:500,:);
test_data = features(4001:4500,:);
test_labels = classlabels(4001:4500,:);


%Initialize variables to store
predicted_labels=zeros(size(test_data,1),1);
ed=zeros(size(test_data,1),size(data,1)); %(MxN) euclidean distances 
ind=zeros(size(test_data,1),size(data,1)); %corresponding indices (MxN)
k_nn=zeros(size(test_data,1),k); %k-nearest neighbors for testing sample (Mxk)

%next calculate the euclidean distance from the test data to the training data

for test_point = 1:size(test_data,1);
  for train_point = 1:size(data,1);
    %compute the euclidean distances and store them in the matrix ed
    ed(test_point,train_point)= sqrt(sum((test_data(test_point,:) - data(train_point,:)).^2));
    
  endfor
      %appends the euclidean distance to the ed matrix and the corresponding 
        %data point index numberninto the index matrix and then sorts the two. 
        %This is a good way to keep track of them.
   [ed(test_point,:), ind(test_point,:)] = sort(ed(test_point,:));
endfor

for k = 1:80
  %find the nearest k for each data point of the testing data
  k_nn = ind(:,1:k);
  nn_index = k_nn(:,1);
  %get the majority vote 
  for i=1:size(k_nn,1)
    %shows the different classes available to choose from
      options=unique(labels(k_nn(i,:)'));
      max_count=0;
      max_label=0;
      %runs this portion of code for each of the feature labels available
      %In this case it is either 1 or 2 but It could be n dimensions 
      for j=1:length(options)
        %finds the indices for each type of label
          L=length(find(labels(k_nn(i,:)')==options(j)));
          J=labels(k_nn(i,:)')==options(j);
          (labels(k_nn(i,:)')==options(j));
    
          if L>max_count
              max_label=options(j);
              max_count=L;
          end
      end
      predicted_labels(i)=max_label;
  end



  %calculate the classification accuracy
  if isempty(test_labels)==0
    %Createsa an array that stores all the labeles that are the same as teh test
    %labels then it devides by the etire test data to get a preportion correct
      accuracy=length(find(predicted_labels==test_labels))/size(test_data,1);
  end
  figure(3)
  hold on 
  scatter(k,accuracy)
end
%as you increase the k nearest neighbors from 1 to 80 the accuracy certainly decreases
%code modeled from (https://www.mathworks.com/matlabcentral/fileexchange/63621-knn-classifier)