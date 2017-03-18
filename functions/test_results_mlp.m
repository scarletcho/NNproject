function [test_results, performance, performance_age] = test_results_mlp(type, test_x, test_y,...
    weights, biases, output_activationF)

% 2016-06-24
% Yejin Cho (scarletcho@korea.ac.kr)

%% Test results (performance) in multi-layer perceptron

switch type
    case 'classification'
        for sample = 1:size(test_x,2)
            
            % Feedforward with the optimized parameters (weights and biases)
            [test_a_sample] = feedforward_mlp(test_x(:,sample), ...
                weights, biases, output_activationF, 'no', 1);
            
            test_a_sample = test_a_sample{1,end};
            test_predictions = find(test_a_sample==max(test_a_sample));
            % [NOTE] In cases where there is multiple maximum values found,
            %        this simply takes the first number in the array.
            
            test_results{1,sample} = [find(test_y(:,sample)==1), test_predictions(1)];
            test_results{2,sample} = arrayfun(@isequal, ...
                test_results{1,sample}(:,1), test_results{1,sample}(:,2));
        end
        
        % Performance measure: classification accuracy
        performance = sum(cell2mat(test_results(2,:)))/size(test_x,2)*100;
        fprintf('accuracy: %.4g %%\n', performance)
        
        
    case 'regression'
        for sample = 1:size(test_x,2)
            % Feedforward with the optimized parameters (weights and biases)
            [test_a_sample] = feedforward_mlp(test_x(:,sample), ...
                weights, biases, output_activationF, 'no', 1);
            test_a_sample = test_a_sample{1,end};
            test_results{1,sample} = test_a_sample;
        end
       
        test_results = cell2mat(test_results);
        
        % Performance measure: MSE (Mean Squared Error)
        performance = mean((test_results - test_y).^2, 2);
        performance_age = mean(mean(abs(test_results - test_y),2)*100);
        fprintf('MSE: %.4g, Error in Age: %.3g\n', performance, performance_age)
        
end
end