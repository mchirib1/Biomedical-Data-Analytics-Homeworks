
function [predicted_labels,nn_index,accuracy] = knnTest(k,data,labels,t_data,t_labels)
  %initialization
predicted_labels=zeros(size(t_data,1),1);
ed=zeros(size(t_data,1),size(data,1)); %ed: (MxN) euclidean distances 
ind=zeros(size(t_data,1),size(data,1)); %corresponding indices (MxN)
k_nn=zeros(size(t_data,1),k); %k-nearest neighbors for testing sample (Mxk)

%calc euclidean distances between each testing data point and the training
%data samples
for test_point=1:size(t_data,1)
    for train_point=1:size(data,1)
        %calc and store sorted euclidean distances with corresponding indices
        ed(test_point,train_point)=sqrt(...
            sum((t_data(test_point,:)-data(train_point,:)).^2));
    end
    [ed(test_point,:),ind(test_point,:)]=sort(ed(test_point,:));
end

%find the nearest k for each data point of the testing data
k_nn=ind(:,1:k);
nn_index=k_nn(:,1);
%get the majority vote 
for i=1:size(k_nn,1)
    options=unique(labels(k_nn(i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(labels(k_nn(i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    predicted_labels(i)=max_label;
end



%calculate the classification accuracy
if isempty(t_labels)==0
    accuracy=length(find(predicted_labels==t_labels))/size(t_data,1);
end



endfunction

[m,n,o] = knn(1,features(1:500,:),classlabels(1:500,:),features(4001:4500),classlabels(4001:4500))
k = 1
data = features(1:500,:)
labels = classlabels(1:500,:)
t_data = features(4001:4500)
t_labels = classlabels(4001:4500)
