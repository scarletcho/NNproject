function [out] = shuffle(in, type)
% "Random order shuffling of an array (or a matrix)."
% "If the input is a matrix, the order of rows is shuffled."

% in: input data (numeric array or matrix)
% type: shuffling type (row / column / all)
%       specify 'row' to shuffle by rows
%               'column' to shuffle by columns
%               'all' to shuffle all the elements

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% shuffle
if iscell(in)
    in = cell2mat(in);
end

% pre-allocation
rnd_index = zeros(size(in));
out = zeros(size(in));

%% shuffle by type: row / column / all
switch type
    case 'row'
        % [option 1] shuffle by rows
        rnd_index = randperm(numel(in(:,1)));
        out = in(rnd_index,:);
        
    case 'column'
        % [option 2] shuffle by columns
        rnd_index = randperm(numel(in(1,:)));
        out = in(:,rnd_index);
        
    case 'all'
        % [option 3] shuffle all elements by columns
        for i = 1:size(in,2)
            rnd_index(:,i) = randperm(numel(in(:,i)));
            out(:,i) = in(rnd_index(:,i),i);
        end
        
end
end