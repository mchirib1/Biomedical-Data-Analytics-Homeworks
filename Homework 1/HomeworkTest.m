#Here are the parameters for the knn function
k = 50;
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
for k = 1:50
  for test_point = 1:size(test_data,1);
    for train_point = 1:size(data,1);
      %compute the euclidean distances and store them in the matrix ed
      ed(test_point,train_point)= sqrt(sum((test_data(test_point,:) - data(train_point,:)).^2));
      
    endfor
    [ed(test_point,:),ind(test_point,:)] = sort(ed(test_point,:));
  endfor


k_nn=ind(:,1:k)
nn_index=k_nn(:,1);
endfor
