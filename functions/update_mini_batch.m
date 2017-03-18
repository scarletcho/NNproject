function [weights, biases, gradient_w, gradient_b] = ...
    update_mini_batch(mini_x_singlebatch, mini_y_singlebatch, weights, biases, eta, costfunction)

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% Update mini-batch via 4 steps
%  >> (1) feedforward
%  >> (2) output error (delta) & gradient calculation
%  >> (3) backpropagtion (if n_layer > 2)
%  >> (4) weight & bias update

%% (1) feedforward
[a, z] = feedforward(mini_x_singlebatch, weights, biases);

%% (2) output error (delta) & gradient calculation
delta = cost_derivative(a, mini_y_singlebatch, costfunction) .* sigmoid_prime(z);
gradient_b = mean(delta,2)/size(delta,2);
gradient_w = mini_x_singlebatch*delta'/size(delta,2);

%----------------------------------------------------------------------------
% cf. Sample-wise (element-wise) calculation
% for sample = 1:mini_batch_size
%     
%     % feedforward
%     a1 = mini_x_singlebatch(:,sample); % input as activation1
%     [a, z] = feedforward(a1, weights, biases);
%     z_stack{sample} = z;
%     a_stack{sample} = a;
%     
%     % output error (delta) & gradient calculation
%     delta(:,sample) = ...
%         cost_derivative(a_stack{sample}, mini_y_singlebatch(:,sample)) ...
%         .* sigmoid_prime(z_stack{sample});
%     
%     gradient_b{sample} = delta(:,sample);
%     gradient_w{sample} = a1*delta(:,sample)';
%     
% end
%
% % average by samples
% gradient_w_mean = average_by_sample(gradient_w);
% gradient_b_mean = average_by_sample(gradient_b);
%----------------------------------------------------------------------------

%% (3) Backpropagation
% for x, y in mini_batch:
%             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
%             nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
%             nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

%backprop(mini_x_singlebatch, mini_y_singlebatch)


%% (4) weight & bias update
weights = weights - eta*gradient_w;
biases = biases - eta*gradient_b;

end