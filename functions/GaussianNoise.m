function [newX, newY] = GaussianNoise(fullX, fullY, multiple)

% 2016-06-24
% Yejin Cho (scarletcho@korea.ac.kr)

%%
ageband = floor(10*fullY)*10;   % age band (10s, 20s, ...)
param = NaiveBayesClass(fullX,ageband);

% Resampling
% (1) Decide the number of new samples in each ageband
nNew = multiple* mean(param.class_dist) - param.class_dist;
for iband = 1:size(nNew,2)      % for 8 age bands
    if nNew(iband) < 0
        nNew(iband) = 0;
    else
        nNew(iband) = ceil(nNew(iband));
    end
end
dataByAge = [ageband ; 100*fullY ; fullX];


% (2) Resample by adding Gaussian Noises
newX_total = [];

for iband = 1:size(nNew,2)  % for 8 age bands (10s, 20s,..)
    
    iAgeInt = find(dataByAge(1,:) == 10*iband); % Indices of ages in an age band
    nAgeInt = numel(iAgeInt);                   % Number of ages in an age band
    nPerAge = round(nNew(iband)/nAgeInt);       % How many samples for each specific age
    
    
    if nPerAge > 0  % if the number of samples lacks!
        newX_doublestack = [];
        
        % for Number of specific ages in an age band
        for iAge = 1:nAgeInt
            
            newX_stack = [];
            % resampling by adding Gaussian noise to all 4005 feature
            for iPerAge = 1:nPerAge
                newX = fullX(:,iAgeInt(iAge)) + randn(4005,1)*.01;
                newX_stack = [newX_stack, newX];
                clear newX
            end
            
            % stack age (newY) with random noise
            newY_stack = repmat(fullY(iAgeInt(iAge))*100, [1, nPerAge]) ...
                + randn(1, nPerAge);
            
            % stack the new age (newY) & data (newX)
            newX_stack = [newY_stack ; newX_stack];
            
            newX_doublestack = [newX_doublestack, newX_stack];
            clear newX_stack
        end
        
        newX_total = [newX_total, newX_doublestack];
        
    end
end

fprintf('Gaussian noise addition is completed\n')

%% output
newX = newX_total(2:end,:);     % features (fMRI data)
newY = newX_total(1,:)*.01;     % label    (age)
end
