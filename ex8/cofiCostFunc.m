function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%C = sum(X.*Theta,2);
C = X * Theta';

D = (C - Y).^2;

E = (C - Y);

J = 1/2*sum(sum(R .* D));

%fprintf('size of E:  %f\n', size(E));
%fprintf('size of R:  %f\n', size(R));

for i = 1:size(X),
	for k = 1:size(X,2),
		sum3 = 0.0;
		for j = 1:size(Theta),
			if R(i,j) == 1
				;X_grad(i,k) = (sum(Theta(j, :) .* X(i, :)) - Y(i,j)) * Theta(j,k);
				sum3 += (sum(Theta(j, :) .* X(i, :)) - Y(i,j)) * Theta(j,k);
			endif;
		end;
		X_grad(i,k) = sum3;
	end;
end;


for j = 1:size(Theta),
	for k = 1:size(Theta,2),
		sum4 = 0.0;
		for i = 1:size(X),
			if R(i,j) == 1
				%Theta_grad(j,k) = (sum(Theta(j, :) .* X(i, :)) - Y(i,j)) * X(i,k);
				sum4 += (sum(Theta(j, :) .* X(i, :)) - Y(i,j)) * X(i,k);
			endif;
		end;
		Theta_grad(j,k) = sum4;
	end;
end;



















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
