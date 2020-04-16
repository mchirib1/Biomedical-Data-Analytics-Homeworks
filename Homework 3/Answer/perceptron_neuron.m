% learning of a perceptron with step function 
%
% X - features
% Y - labels
% theta_init - initial values of weights
% niter - number of iterations
% learningRate - learning rate
%
% theta - return the learned weights
%
% qwei

function theta = perceptron_neuron(X,Y,theta_init,niter,learningRate)
  num_examples = length(X);
  theta = theta_init;
  
  %Iterate until some critera is met
  for j = 1:niter;
    %Sequentially update by iterating over the training examples
    for i = 1:num_examples;
      
      %The neuron potential can be calculated in 2 ways
      %summation of weight value products or dot product with theta 
      activation(i,:) = sum(X(i,:).*theta);
      
      %If the activation breaks threshold 0, predict label 1
      if activation(i,:)>=0;
        predicted_label(i,:) = 1;
        
        %If activation does not break threshold of 1, predict label -1
      else 
        predicted_label(i,:) = -1;
      end
      
      %if predicted label does not equal the training label update theta
      if predicted_label(i,:)~=Y(i,:);
        %first calculate the change in theta
        delta_theta = learningRate*(Y(i,:)-predicted_label(i,:))*X(i,:);
        
        %calculate the new theta based on 
        theta = delta_theta + theta;
      end
    end
    theta;
    j;
  end
end