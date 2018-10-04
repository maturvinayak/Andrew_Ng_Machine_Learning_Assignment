function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

dummy_theta = theta;
dummy_theta(1) = 0;
h = X * theta;
h_new = sigmoid(h);
J = ((-1/m) * (sum(y.*log(h_new)+(1-y).*log(1-h_new))))+(lambda/(2*m))*sum(dummy_theta.^2);
grad = ((1/m)*X'*(h_new-y)) + ((lambda/m)*dummy_theta);




% =============================================================

end
