function [newX, newY] = NaiveBayesSampling(fullX, fullY, multiple)

% 2016-06-24
% Yejin Cho (scarletcho@korea.ac.kr)

%%
close all;
ageband = floor(10*fullY)*10;   % age band (10s, 20s, ...)
ageRange = 18:85;               % age range in integer

param = NaiveBayesClass(fullX,ageband);
clear x0 y0

%% Polynomial fitting on the means of each 4005 feature
%   from each ageband (Each mean (=fy) is coupled with
%   an the average age in the ageband (=fx).)

for ifeat = 1:size(fullX,1)
    fmu = param.mu(ifeat,1:7);
    fsig = param.sigma(ifeat,1);
    
    idxAgeband = [];
    meanXinAgeband = [];
    
    for k = 1:7
        % [NOTE] The single sample in the 80s is excluded
        %        for better generalization
        idxAgeband{1,k} = find(ageband == param.unique_class(k));
        meanXinAgeband(1,k) = mean(fullY(idxAgeband{1,k}));
    end
    
    %% 4th degree Polynomial fitting
    % (cf. ill-conditioned when higher degree is applied)
    fx = 100*meanXinAgeband;
    fy = fmu;
    
    fp = polyfit(fx, fy, 4);
    interpolatedMean(ifeat,:) = polyval(fp, ageRange);
    
    %% Uncomment the following 3 lines to see the plots:
    plot(fx, fy,'o--'); hold on
    plot(18:85, interpolatedMean(ifeat,:), 'r'); hold off
    shg; pause; clf;
    close all
end
meanByAge = [floor(ageRange*.1)*10 ; ageRange ; interpolatedMean];


%% Resampling
% (1) Decide the number of new samples in each ageband
%     -> sample upto N multiple(s) of the average number of
%        samples in the distribution
%        (cf. avg == 29 samples in each age band)

nNew = multiple * mean(param.class_dist) - param.class_dist;

for iband = 1:size(nNew,2)      % for 8 age bands
    if nNew(iband) < 0
        nNew(iband) = 0;
    else
        nNew(iband) = ceil(nNew(iband));
    end
end


% (2) Resample based on the interpolated mu & sigma of each age
newY_total = [];

for iband = 1:size(nNew,2)  % for 8 age bands (10s, 20s,..)
    
    newY_stack = [];
    iAgeInt = find(meanByAge(1,:) == 10*iband); % Indices of ages in an age band
    nAgeInt = numel(iAgeInt);                   % Number of ages in an age band
    nPerAge = round(nNew(iband)/nAgeInt);       % How many samples for each specific age
    
    
    if nPerAge > 0  % if the number of samples lacks!
        
        % for Number of specific ages in an age band
        for iAge = 1:nAgeInt
            fmuAge = interpolatedMean(:,iAgeInt(iAge));
            
            % resampling of every 4,005 feature
            for ifeat = 1:size(fmuAge,1)
                newY(ifeat,:) = normrnd(fmuAge(ifeat), param.sigma(ifeat), ...
                    [1, nPerAge]);
            end
            
            newY = [repmat(ageRange(iAgeInt(iAge)), [1, nPerAge]) ; newY];
            newY_stack = [newY_stack, newY];
            clear newY
        
        end 
        newY_total = [newY_total, newY_stack];
        
    end 
end

fprintf('Bayesian resampling is completed\n')

%% output
newX = newY_total(2:end,:);     % features (fMRI data)
newY = newY_total(1,:)*.01;     % label    (age)
end
