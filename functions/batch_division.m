function [mini_x, mini_y] = batch_division(x, y, batch_size)
% "Mini-batch division function."
% n: number of training samples (total)
% m: number of mini-batches
% batch_size: number of samples per mini-batch
% mini_training & mini_test: (batch_size) by (m) matrix

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% mini-batch division
n = size(x,2);
batch_idx = 1:(batch_size):n;

for m = 1:numel(batch_idx);
    batch_end_idx = batch_idx(m)+batch_size-1;
    if n >= batch_end_idx
        mini_x{1,m} = x(:,batch_idx(m):batch_end_idx);
        mini_y{1,m} = y(:,batch_idx(m):batch_end_idx);
    end
end
end