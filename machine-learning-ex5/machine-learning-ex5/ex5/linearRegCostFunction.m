function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%[temp a b]=featureNormalize(X(:,2));
%[y c d]=featureNormalize(y);
%X(:,2)=temp

h=X*theta;

J_unreg= (1/2/m)*((h-y))'*((h-y));

J_reg=J_unreg + (lambda/2/m)*(theta'*theta-theta(1)^2);
J=J_reg;
grad=1/m*X'*(h-y);
grad=grad+lambda/m*theta;
grad(1)=grad(1)-lambda/m*theta(1);










% =========================================================================

grad = grad(:);

end
