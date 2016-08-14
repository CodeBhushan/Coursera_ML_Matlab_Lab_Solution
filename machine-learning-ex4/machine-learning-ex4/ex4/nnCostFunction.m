function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X=[ones(size(X,1),1),X];
a2=(Theta1*X')';
a2=sigmoid(a2);
a2=[ones(size(a2,1),1),a2];
a3=(Theta2*a2')';
a3=sigmoid(a3);
hTheta=a3;
%c=[1,2,3,4,5,6,7,8,9,10];
c=1:num_labels;
y_m_c=zeros(m,num_labels);
for i=1:m
    y_m_c(i,:)=(y(i)==c);
end
size(y_m_c);
J=(1/m)*(-1*( sum(sum(y_m_c.*log(hTheta),2))  )-   sum(sum((1-y_m_c).*log(1-hTheta),2))     );
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
for t=1:m
    a1=X(t,:);
    z2=(Theta1*a1')';
    a2=sigmoid(z2);
    a2=[ones(size(a2,1),1),a2];
    z3=(Theta2*a2')';
    a3=sigmoid(z3);
    %del for output layer
    del3=zeros(num_labels,1);
    %for out=1:num_labels
     %   del3(out)=a3(out)-y(t,out);
    %end
    del3=a3-y_m_c(t,:);
    
    
    
        %del for hidden layer i.e. sencod layer
        del2=zeros(1,1+hidden_layer_size);
        del2= ((Theta2')*del3')';
        del2=del2(2:end);
        del2=del2.*sigmoidGradient(z2);
        Theta2_grad=Theta2_grad+del3'*a2;
        Theta1_grad=Theta1_grad+del2'*a1;
end
Theta2_grad=Theta2_grad./m;
Theta1_grad=Theta1_grad./m;
            
   
    
    
 
%grad=(1/m)*(X'*(hTheta-y)    )';
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
J=J+(lambda/2/m)*((  sum(sum(Theta1.^2,2))+   sum(sum(Theta2.^2,2))           )-(Theta1(:,1)'*Theta1(:,1) + Theta2(:,1)'*Theta2(:,1))          );




%% 

% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)


%  J=J+(lambda/2/m)*(sum((all_theta_unrolled.^2)) - theta(:,1)'*theta(:,1));
%  temp=theta;
%  temp(:,1)=zeros(size(theta,1),1);
%  grad=grad+(lambda/m)*temp;

tempTheta1=Theta1;
tempTheta1(:,1)=zeros(size(tempTheta1,1),1);
tempTheta1=tempTheta1*lambda/m;
Theta1_grad=Theta1_grad+tempTheta1;
tempTheta2=Theta2;
tempTheta2(:,1)=zeros(size(tempTheta2,1),1);
tempTheta2=tempTheta2*lambda/m;
Theta2_grad=Theta2_grad+tempTheta2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
