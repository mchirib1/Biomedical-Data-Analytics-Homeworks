% learning a logistic neuron with sigmoid transfer function 
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

function theta = logistic_neuron(X,Y,theta_init,niter,learningRate)
%Similar to the perceptron the iterations and theta need to be defined
num_examples = size(X,1);
theta = theta_init;

%Iterate until a criteria is met
for j = 1:niter
    %Iterate through the number of examples
    for i = 1:num_examples 
        
        %compute the activation
        %activation(i,:) = X(i,:)*theta';
        %Convert to nonlinear activation with the sigmoid function
        s(i,:) = sig(X(i,:))*theta';
        %Compute teh derivative of the sigmoid for gradient descent
        ds(i,:) = dsig( s(i,:));
        
      %If the activation breaks threshold 0, predict label 1
      if s(i,:)>= 0;
        predicted_label(i,:) = 1;
        
        %If activation does not break threshold of 1, predict label -1
      else 
        predicted_label(i,:) = -1;
      end
      
      %if predicted label does not equal the training label update theta
      if predicted_label(i,:)~=Y(i,:);
        %first calculate the change in theta
        delta_theta = learningRate*(Y(i,:)-predicted_label(i,:))*ds(i,:)*X(i,:);
        
        %calculate the new theta based on 
        theta = delta_theta + theta;
        
      end
     theta;
     
     j;
    end
end