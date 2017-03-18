function mean_by_samples = average_by_sample(cell_by_samples)

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% mean by samples
sum_by_samples = zeros(size(cell_by_samples{1,1}));
for sample = 1:size(cell_by_samples,2)
    sum_by_samples = sum_by_samples + cell_by_samples{1,sample};
end
mean_by_samples = sum_by_samples/size(cell_by_samples,2);
end