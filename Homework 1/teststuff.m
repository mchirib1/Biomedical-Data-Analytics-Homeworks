[m,n,o] = knn(1,features(1:500,:),classlabels(1:500,:),features(4001:4500),classlabels(4001:4500))
k = 1
data = features(1:500,:)
labels = classlabels(1:500,:)
t_data = features(4001:4500)
t_labels = lasslabels(4001:4500)