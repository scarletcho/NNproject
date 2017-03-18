%% ANN project: Neural Network design for Age prediction
%  by. Yejin Cho (scarletcho@korea.ac.kr)
%  Last updated: 2016-06-25
clc;clear all;close all;
lambda_opt = [];
error_opt = [];
%% *** PLEASE CHOOSE YOUR MODEL SETTING:
% a. k-fold cross-validation: (nfolds)
%    (= without inner CV for parameter optimization)
    nfolds = 4;
    nestedcv = 'no';    % 'yes' if nested setting desired

% b. k by v NESTED cross-validation: (nfolds) by (vfolds)
    vfolds = 2;         % 5 by 2 nested cross-validation

% c. Regularization option: L1, L2
    regularization_set = {'none'};    % 'none', 'L1', 'L2', or 'L1L2'
    switch regularization_set{1}
        case 'none'
            lambda_set = 0;
    end
    
% d. Dropout option
    dropout = 'no';  % 'yes' or 'no'
    dropoutRate = 1; % 0.5;
    
% e. Resampling option
    resample = 'Bayesian';  % 'none', 'Bayesian', or 'GaussianNoise'
    multiple = 2;  % target number of samples
                   % (=> N multiple(s) of average samples per ageband;
                   %  => N * 29 (avg).)
                   
                   % e.g. multiple = 2;
                   % = Make the DOUBLE of the avg. number of samples per ageband
                   % => makes 2*avg N samples per group; 29*2 = 54) 
    
%% ----------------------------------------------------------------------
%                 Choice of COST function & Task
%------------------------------------------------------------------------
% a.'MSE'           : quadratic cost; Mean Squared Error (actF: sigmoid)
% b.'cross_entropy' : cross-entropy (actF: sigmoid)
%------------------------------------------------------------------------
cost_function = 'MSE';
task = 'regression';    % Task type: 'classification' or 'regression'
%------------------------------------------------------------------------
% Hyperparameter setting FIXED for the chosen cost function
switch cost_function        
    case 'cross_entropy'
        eta_set2 = 0.2;
        n_hidden_set = 50;
        minibatch_size_set = 7;
        n_epochs_set = 30;
        output_activationF = 'sigmoid';
        
    case 'MSE'
        eta_set = 0.5;
        n_hidden_set = 90;
        minibatch_size_set = 20;
        n_epochs_set = 120;
        output_activationF = 'sigmoid';
        
end

%% ----------------------------------------------------------------------
%                     Data loading & Normalization
%------------------------------------------------------------------------
% Load data
rng('default')	% put the setting of random number generators to default
addpath('./functions')
load data.mat

fullX = cell2mat(x0')';
fullY = cell2mat(y0')';

clear x0 y0
%% [NOTE] The given data (x,y) is re-organized into:
%         (FEATURES) by (SAMPLES)
%% ----------------------------------------------------------------------
%                   (Outer) K-fold sample partition
%------------------------------------------------------------------------
nidx = kfold_partition(fullX, nfolds);  % get indices for partition

%% ----------------------------------------------------------------------
%              Beginning of the Outer cross-validation loop
%------------------------------------------------------------------------
optimized_performance_table = table();

for ifold_out = 1:nfolds % outer CV loop (1 ~ nfolds)
    clear testRange trainRange train_x train_y test_x test_y
    
    testRange = nidx(ifold_out,1):nidx(ifold_out,2);
    trainRange = setdiff(1:size(fullX,2), testRange);

    train_x = fullX(:, trainRange);
    train_y = fullY(:, trainRange);
    test_x = fullX(:, testRange);
    test_y = fullY(:, testRange);
    
    
    switch resample
        case 'Bayesian'
            [newX, newY] = NaiveBayesSampling(train_x,train_y,multiple);
            % [NOTE] Please switch train_x,train_y into fullX,fullY
            %        if resampling from the full data is desired.
            train_x = [train_x, newX];
            train_y = [train_y, newY];
            
        case 'GaussianNoise'
            [newX, newY] = GaussianNoise(train_x,train_y,multiple);
            % [NOTE] Please switch train_x,train_y into fullX,fullY
            %        if resampling from the full data is desired.
            train_x = [train_x, newX];
            train_y = [train_y, newY];
            
    end
    
    %------------------------------------------------------------------------
    %                Hyperparameter CANDIDATES to be optimized
    %------------------------------------------------------------------------
    lambda_set = [0.1 0.01 0.001 0.0001 0.00001];   % 0.1;
    switch regularization_set{1}
        case 'none'
            lambda_set = 0;
    end
    
    switch nestedcv
        
        % (Start from here, if nested CV wanted)
        case 'yes'
            % zero pre-allocation for validation performance record
            val_performance_record = zeros(numel(lambda_set), vfolds);
            
            for lambdanum = 1:numel(lambda_set)
                regnum = numel(regularization_set);
                epochnum = numel(n_epochs_set);
                etanum = numel(eta_set);
                hnum = numel(n_hidden_set);
                batchnum = numel(minibatch_size_set);
                %% ----------------------------------------------------------------------
                %                    Inner V-fold sample partition
                %------------------------------------------------------------------------
                vfolds = 2;
                vidx = kfold_partition(train_x, vfolds);
                
                for ifold_inner = 1:vfolds
                    clear valRange realtrRange realtr_x realtr_y val_x val_y
                    valRange = vidx(ifold_inner,1):vidx(ifold_inner,2);
                    realtrRange = setdiff(1:size(train_x,2), valRange);
                    
                    realtr_x = train_x(:, realtrRange);
                    realtr_y = train_y(:, realtrRange);
                    val_x = train_x(:, valRange);
                    val_y = train_y(:, valRange);
                    
                    %% ----------------------------------------------------------------------
                    %                     [ Neural Network Training ]
                    %------------------------------------------------------------------------
                    %                     1. Initial parameter set-up
                    %------------------------------------------------------------------------
                    % Hyperparameter setting (from the given candidates)
                    lambda = lambda_set(lambdanum);
                    regularization = regularization_set{regnum};
                    n_epochs = n_epochs_set(epochnum);
                    eta = eta_set(etanum);
                    n_hidden = n_hidden_set(hnum);
                    
                    netsize = [size(realtr_x,1) n_hidden size(realtr_y,1)];
                    n_layers = size(netsize,2);
                    
                    % weight & bias initialization
                    biases = {normrnd(0,1,netsize(2),1)*.01, normrnd(0,1,netsize(3),1)*.01};
                    weights = {normrnd(0,1/sqrt(size(realtr_x,1)),netsize(1),netsize(2))*.01, ...
                        normrnd(0,1/sqrt(size(realtr_x,1)),netsize(2),netsize(3))*.01};
                    
                    %% ----------------------------------------------------------------------
                    %                     2. Stochastic Gradient Descent
                    %------------------------------------------------------------------------
                    %% The 1st loop in SGD: EPOCHs
                    for epochs = 1:n_epochs
                        training_data = shuffle([realtr_x ; realtr_y], 'column');
                        realtr_x = training_data(1:size(realtr_x,1),:);
                        realtr_y = training_data(size(realtr_x,1)+1:end,:);
                        
                        minibatch_size = minibatch_size_set(batchnum);
                        [mini_x, mini_y] = batch_division(realtr_x, realtr_y, minibatch_size);
                        
                        n_minibatch = size(mini_x,2);
                        
                        % [NOTE] mini_x & mini_y: 1 by m cell (cf. m: number of mini-batches)
                        %        Each cell: (nodes) by (minibatch_size)
                        
                        %--------------------------------------------------------------------
                        %% The 2nd loop in SGD: MINI-BATCHES
                        for minibatch = 1:n_minibatch	% split minibatches
                            mini_x_singlebatch = mini_x {1,minibatch};
                            mini_y_singlebatch = mini_y {1,minibatch};
                            clear a_next z_next delta stack_a stack_z
                            %----------------------------------------------------------------
                            %% The 3rd loop in SGD: LAYERs
                            [stack_a, stack_z, stack_mask] = feedforward_mlp(mini_x_singlebatch,...
                                weights, biases, output_activationF, dropout, dropoutRate);
                            
                            % (1) final-DELTA computation: output error of the final layer (BP1)
                            switch cost_function
                                case 'MSE'
                                    delta{1,1} = ...
                                        cost_derivative(stack_a{1,end}, mini_y_singlebatch, cost_function)...
                                        .* sigmoid_prime(stack_z{1,end});
                                    
                                case {'cross_entropy', 'neg_log'}
                                    delta{1,1} = stack_a{1,end} - mini_y_singlebatch;
                            end
                            %----------------------------------------------------------------
                            % (2) BACKPROPAGATION (BP2) of later deltas toward the earlier layers
                            for layer = size(netsize,2)-1:-1:2
                                switch dropout
                                    case 'yes'
                                        delta = [delta, ...
                                             (weights{layer} * delta{1,1} ...
                                             .* sigmoid_prime(stack_z{1,layer-1}))...
                                             .* stack_mask{1,layer-1}];

                                    case 'no'
                                        delta = [delta, weights{layer} * delta{1,1} .* sigmoid_prime(stack_z{1,layer-1})];
                                        
                                end
                            end % (the end of layer loop)
                            
                            % cf. Flip the matrix so that deltas from earlier layers are located in earlier columns
                            delta = fliplr(delta);
                            
                            %----------------------------------------------------------------
                            % (3) GRADIENT computation: gradient of COST func w.r.t. biases & weights
                            % 3-(i). gd of w & b: in the 1st layer
                            gradient_w{1} = mini_x_singlebatch*delta{1}'/minibatch_size;
                            gradient_b{1} = mean(delta{1},2)/minibatch_size;

                            
                            % 3-(ii). gd of w & b: in the 2nd to (final-1)th layer
                            for layer = 2:n_layers-1
                                gradient_w{layer} = stack_a{1,layer-1}*delta{1,layer}'/minibatch_size;
                                gradient_b{layer} = mean(delta{1,layer},2)/minibatch_size;
                                
                                %------------------------------------------------------------
                                %       [ REGULARIZATION term addition (L1 or L2) ]
                                %------------------------------------------------------------
                                switch regularization
                                    case 'none'
                                        gradient_w{layer} = gradient_w{layer};
                                        
                                    case 'L1'
                                        gradient_w{layer} = gradient_w{layer} ...
                                            + lambda/size(stack_a{1,end},2)*sign(weights{layer});
                                        
                                    case 'L2'
                                        gradient_w{layer} = gradient_w{layer} ...
                                            + lambda/size(stack_a{1,end},2)*weights{layer};
                                    
                                    case 'L1L2'
                                        gradient_w{layer} = gradient_w{layer} ...
                                            + lambda/size(stack_a{1,end},2)*sign(weights{layer}) ...
                                            + lambda/size(stack_a{1,end},2)*weights{layer};
                                                 
                                end
                                %------------------------------------------------------------
                            end % (the end of layer loop)
                            
                            % 3-(iii). UPDATE weight & bias: from the 1st to (final-1)th layer
                            for layer = 1:n_layers-1
                                weights{1,layer} = weights{1,layer} - eta*gradient_w{1,layer};
                                biases{1,layer} = biases{1,layer} - eta*gradient_b{1,layer};
                            end % (the end of layer loop)
                            
                        end % (the end of minibatch loop)
                        
                        % This is the END of training!
                        %--------------------------------------------------------------------
                        fprintf('Epoch %d / %d completed\n', epochs, n_epochs)
                        test_results_mlp(task, val_x, val_y, weights, biases, output_activationF);
                        
                    end % (the end of epoch loop)
                    
                    %% ----------------------------------------------------------------------
                    %                     3. TEST results
                    %------------------------------------------------------------------------
                    fprintf('Validation results:')
                    clear val_results accuracy
                    
                    
                    switch dropout
                        case 'yes'
                            % Compensation for the weights trained from the dropout setting
                            for ilayer = 1:numel(weights)
                                weights{1,ilayer} = ...
                                    weights{1,ilayer} * (1-dropoutRate); % mean(mean(mask));
                            end
                    end
                    
                    [~, val_performance] = test_results_mlp(task, val_x, val_y, ...
                                           weights, biases, output_activationF);
                    
                    val_performance_record(lambdanum, ifold_inner) = val_performance;
                    param_performance_table = table(lambda_set', val_performance_record(:,1),...
                        val_performance_record(:,2), 'VariableNames',...
                        {'lambda', 'innerfold1', 'innerfold2'});
                    
                    % Leave records on a text file
                    reports = ['\nresult ' num2str(val_performance) ...
                        ' batchsize ' num2str(minibatch_size) ...
                        ' eta ' num2str(eta) ...
                        ' epochs ' num2str(n_epochs) ...
                        ' hiddens ' num2str(n_hidden) ...
                        ' costF ' cost_function ...
                        ' reg ' regularization ...
                        ' lambda ' num2str(lambda) ...
                        ' dropout ' num2str(dropoutRate) ...
                        ' resample ' resample ...
                        ' multiple ' num2str(multiple)];
                    
                    fid = fopen('nestedCV_report.txt','a');
                    fprintf(fid, reports);
                    fclose(fid);
                    
                    %------------------------------------------------------------------------
                end % (the end of inner CV loop)
            end % (the end of parameter loop)
            
            mean_param_performance = mean(val_performance_record,2);
            iparam_optimal = find(mean_param_performance==min(mean_param_performance));
            
            param_performance_table.meanError = mean_param_performance;
            param_performance_table.testError = zeros(height(param_performance_table),1);
            param_performance_table.ageError = zeros(height(param_performance_table),1);
            param_performance_table.testResults = cell(height(param_performance_table),1);
            param_performance_table.testTargets = cell(height(param_performance_table),1);
            
            %% Hereafter, use the optimized lambda!            
            optimized_performance_table = [optimized_performance_table ; ...
                param_performance_table(iparam_optimal,:)];
            
            lambda_set = optimized_performance_table.lambda(...
                find(optimized_performance_table.lambda==...
                min(optimized_performance_table.lambda)));
            
            disp('This is the End of INNER cross-validation (for hyperparameter choice)');
            
    end % (the end of inner CV switch-case)
    
    %% ------------------------------------------------------------------------
    % (Start from here, if NO nested CV wanted;
    %  => cross-validation without parameter optimizing inner validation)
    %------------------------------------------------------------------------
    %     NN Training & Testing
    %     given the optimized hyperparameter
    %------------------------------------------------------------------------
    clear biases delta fid gradient_b gradient_w ifold_inner iparam_optimal ...
          mini_x mini_x_singlebatch mini_y mini_y_singlebatch ...
          realtr_x realtr_y realtrRange reports stack_a stack_z val_x val_y ...
          valRange vidx weights
    %------------------------------------------------------------------------
    %                     [ Neural Network Training ]
    %------------------------------------------------------------------------
    %                     1. Initial parameter set-up
    %------------------------------------------------------------------------
    % Hyperparameter setting (fixed)
    lambdanum = numel(lambda_set);
    regnum = numel(regularization_set);
    epochnum = numel(n_epochs_set);
    etanum = numel(eta_set);
    hnum = numel(n_hidden_set);
    batchnum = numel(minibatch_size_set);
    
    lambda = lambda_set(lambdanum);
    regularization = regularization_set{regnum};
    n_epochs = n_epochs_set(epochnum);
    eta = eta_set(etanum);
    n_hidden = n_hidden_set(hnum);
    
    netsize = [size(train_x,1) n_hidden size(train_y,1)];
    n_layers = size(netsize,2);
    
    % weight & bias initialization
    biases = {normrnd(0,1,netsize(2),1)*.01, normrnd(0,1,netsize(3),1)*.01};
    weights = {normrnd(0,1/sqrt(size(train_x,1)),netsize(1),netsize(2))*.01, ...
        normrnd(0,1/sqrt(size(train_x,1)),netsize(2),netsize(3))*.01};
    
    %% ----------------------------------------------------------------------
    %                     2. Stochastic Gradient Descent
    %------------------------------------------------------------------------
    %% The 1st loop in SGD: EPOCHs
    for epochs = 1:n_epochs
        training_data = shuffle([train_x ; train_y], 'column');
        train_x = training_data(1:size(train_x,1),:);
        train_y = training_data(size(train_x,1)+1:end,:);
        
        minibatch_size = minibatch_size_set(batchnum);
        [mini_x, mini_y] = batch_division(train_x, train_y, minibatch_size);
        
        n_minibatch = size(mini_x,2);
        
        % [NOTE] mini_x & mini_y: 1 by m cell (cf. m: number of mini-batches)
        %        Each cell: (nodes) by (minibatch_size)
        %--------------------------------------------------------------------
        %% The 2nd loop in SGD: MINI-BATCHES
        for minibatch = 1:n_minibatch	% split minibatches
            mini_x_singlebatch = mini_x {1,minibatch};
            mini_y_singlebatch = mini_y {1,minibatch};
            clear a_next z_next delta stack_a stack_z
            %----------------------------------------------------------------
            %% The 3rd loop in SGD: LAYERs
            switch dropout
                case 'no'
                    [stack_a, stack_z, ~] = feedforward_mlp(mini_x_singlebatch,...
                        weights, biases, output_activationF, dropout, dropoutRate);
                case 'yes'
                    [stack_a, stack_z, stack_mask] = feedforward_mlp(mini_x_singlebatch,...
                        weights, biases, output_activationF, dropout, dropoutRate);
            end
            
            % (1) final-DELTA computation: output error of the final layer (BP1)
            switch cost_function
                case 'MSE'
                    delta{1,1} = ...
                        cost_derivative(stack_a{1,end}, mini_y_singlebatch, cost_function)...
                        .* sigmoid_prime(stack_z{1,end});
                    
                case {'cross_entropy', 'neg_log'}
                    delta{1,1} = stack_a{1,end} - mini_y_singlebatch;
            end
            %----------------------------------------------------------------
            % (2) BACKPROPAGATION (BP2) of later deltas toward the earlier layers
            for layer = size(netsize,2)-1:-1:2
                switch dropout
                    case 'yes'
                        delta = [delta, ...
                            (weights{layer} * delta{1,1} ...
                            .* sigmoid_prime(stack_z{1,layer-1}))...
                            .* stack_mask{1,layer-1}];
                        
                    case 'no'
                        delta = [delta, ...
                            weights{layer} * delta{1,1} .* sigmoid_prime(stack_z{1,layer-1})];
                        
                end
            end % (the end of layer loop)
            
            % cf. Flip the matrix so that deltas from earlier layers are located in earlier columns
            delta = fliplr(delta);
            
            %----------------------------------------------------------------
            % (3) GRADIENT computation: gradient of COST func w.r.t. biases & weights
            % 3-(i). gd of w & b: in the 1st layer
            gradient_w{1} = mini_x_singlebatch*delta{1}'/minibatch_size;
            gradient_b{1} = mean(delta{1},2)/minibatch_size;
            
            % 3-(ii). gd of w & b: in the 2nd to (final-1)th layer
            for layer = 2:n_layers-1
                
                gradient_w{layer} = stack_a{1,layer-1}*delta{1,layer}'/minibatch_size;
                gradient_b{layer} = mean(delta{1,layer},2)/minibatch_size;
                
                %------------------------------------------------------------
                %       [ REGULARIZATION term addition (L1 or L2) ]
                %------------------------------------------------------------
                switch regularization
                    case 'none'
                        gradient_w{layer} = gradient_w{layer};
                        
                    case 'L1'
                        gradient_w{layer} = gradient_w{layer} ...
                            + lambda/size(stack_a{1,end},2)*sign(weights{layer});
                        
                    case 'L2'
                        gradient_w{layer} = gradient_w{layer} ...
                            + lambda/size(stack_a{1,end},2)*weights{layer};
                        
                    case 'L1L2'
                        gradient_w{layer} = gradient_w{layer} ...
                            + lambda/size(stack_a{1,end},2)*sign(weights{layer}) ...
                            + lambda/size(stack_a{1,end},2)*weights{layer};
                        
                end
                %------------------------------------------------------------
            end % (the end of layer loop)
            
            % 3-(iii). UPDATE weight & bias: from the 1st to (final-1)th layer
            for layer = 1:n_layers-1
                weights{1,layer} = weights{1,layer} - eta*gradient_w{1,layer};
                biases{1,layer} = biases{1,layer} - eta*gradient_b{1,layer};
            end % (the end of layer loop)
            
        end % (the end of minibatch loop)
        
        % This is the END of training!
        %--------------------------------------------------------------------
        fprintf('Epoch %d / %d completed\n', epochs, n_epochs)
        
        test_results_mlp(task, test_x, test_y, weights, ...
            biases, output_activationF);
        
    end % (the end of epoch loop)
    
    %% ----------------------------------------------------------------------
    %                     3. TEST results
    %------------------------------------------------------------------------
    fprintf('Test results:')
        
    switch dropout
        case 'yes'
            % Compensation for the weights trained from the dropout setting
            for ilayer = 1:numel(weights)
                weights{1,ilayer} = ...
                    weights{1,ilayer} * (1-dropoutRate); % mean(mean(mask));
            end
    end
    
    [test_results, test_performance, performance_age] = test_results_mlp...
        (task, test_x, test_y, weights, biases, output_activationF);
    R = regression(test_results, test_y);
    
    figure, plotregression(test_results, test_y)
    title(sprintf('%d-fold Test result (%dth fold)', nfolds, ifold_out), 'FontSize', 15)
    xlabel(sprintf('MSE: %.2g, Error in Age: %.2f, R: %.3g\n', ...
        test_performance, performance_age, R), 'FontSize', 15)
    drawnow
    savefig(sprintf('MSE%.2gFold%d_%s.fig', ...
        test_performance, ifold_out, dateFormat));
    
    optimized_performance_table.testError(ifold_out,1) = test_performance;
    optimized_performance_table.ageError(ifold_out,1) = performance_age;
    optimized_performance_table.testResults{ifold_out,1} = test_results;
    optimized_performance_table.testTargets{ifold_out,1} = test_y;
    
    % Leave records on a text file
    reports = ['\ndate ' dateFormat, ...
        ' result ' num2str(test_performance), ...
        ' ageError ' num2str(performance_age), ...
        ' R ' num2str(R), ...
        ' batchsize ' num2str(minibatch_size), ...
        ' eta ' num2str(eta), ...
        ' epochs ' num2str(n_epochs), ...
        ' hiddens ' num2str(n_hidden), ...
        ' costF ' cost_function, ...
        ' reg ' regularization, ...
        ' lambda ' num2str(lambda), ...
        ' dropout ' num2str(dropoutRate), ...
        ' resample ' resample ...
        ' multiple ' num2str(multiple)];
    
    fid = fopen('nestedCV_report.txt','a');
    fprintf(fid, reports);
    fclose(fid);
    
    %------------------------------------------------------------------------
end % (the end of outer CV loop)

% Outer CV test accuracy (= Average error from all k cases)
meantestError = mean(optimized_performance_table.testError);
fprintf('Outer CV test result: %.4g %%\n', meantestError)

bestidx = find(optimized_performance_table.testError == ...
    min(optimized_performance_table.testError));
bestResults = optimized_performance_table.testResults(bestidx);
bestTargets = optimized_performance_table.testTargets(bestidx);
R = regression(bestResults, bestTargets);

figure, plotregression(bestResults, bestTargets)
title(sprintf('%d-fold Test result (%dth fold)', nfolds, bestidx), 'FontSize', 15)
xlabel(sprintf('MSE: %.2g, Error in Age: %.2f, R: %.3g', ...
    optimized_performance_table.testError(bestidx),...
    optimized_performance_table.ageError(bestidx), R), 'FontSize', 15)
drawnow

% Leave records on a text file
reports = ['\ndate ' dateFormat, ...
    ' result ' num2str(meantestError), ...
    ' bestErr ' num2str(min(optimized_performance_table.testError)), ...
    ' bestageErr ' num2str(min(optimized_performance_table.ageError)), ...
    ' bestR ' num2str(R), ...
    ' batchsize ' num2str(minibatch_size), ...
    ' eta ' num2str(eta), ...
    ' epochs ' num2str(n_epochs), ...
    ' hiddens ' num2str(n_hidden), ...
    ' costF ' cost_function, ...
    ' reg ' regularization, ...
    ' lambda ' num2str(lambda), ...
    ' dropout ' num2str(dropoutRate), ...
    ' resample ' resample, ...
    ' multiple ' num2str(multiple)];

fid = fopen('outerCV_report.txt','a');
fprintf(fid, reports);
fclose(fid);
