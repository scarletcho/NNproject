function [iPart] = kfold_partition(fulldata, nfolds)
% "K-fold cross-validtion set division function."
% fulldata: full set of data of (features) by (samples)
% nfolds: number of folds in cross-validation
%         (= number of desired partitions)

% 2016-06-20
% Yejin Cho (scarletcho@korea.ac.kr)

%% K-fold sample partition
% (1) Get indices of partition
Nsamples = size(fulldata,2);       % Nsamples == Ncolumns of fulldata
iRef = 0:Nsamples/nfolds:Nsamples; % reference indices
iRef = floor(iRef);

for k = 1:nfolds
    iPart(k,:) = [iRef(k)+1, iRef(k+1)];
end
 
% % (2) Divide x,y into k partitions
% for k = 1:nfolds
%     partitions{1,k} = fulldata(:, iPart(k,1):iPart(k,2));
% end

end