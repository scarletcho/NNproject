function [out] = cost_derivative(output_activations, y, cost_function)
% "Return the vector of partial derivatives
%  partial C_x / partial a for the output activations."

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)
%%
switch cost_function
    case 'MSE'
        out = output_activations - y;
        
    case 'cross_entropy'
        out = (output_activations - y)./...
            (output_activations.*(1-output_activations));
end
% 
% switch regularization
%     case 'L2'
%         out = out + lambda/(2*size(output_activations,2))*sum(weights.^2);
    %case 'L1'        
    
end
