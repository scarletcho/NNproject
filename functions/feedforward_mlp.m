function [stack_a, stack_z, stack_mask] = feedforward_mlp(input, weights, biases, ...
    output_activationF, dropout, dropoutRate)

% 2016-06-16
% Yejin Cho (scarletcho@korea.ac.kr)

%% Feedforward using sigmoid function in multi-layer perceptron
%% Dropout
switch dropout
    % If YES dropout setting
    case 'yes'
        a_prev = input;
        switch output_activationF
            case 'softmax'
                for layer = 1:size(weights,2)-1
                    [a_next, z_next] = feedforward(a_prev, ...
                        weights{layer}, biases{layer}, 'sigmoid');
                    
                    % Create a Masking matrix M for drop-out
                    mask = rand(size(a_next));
                    for idx = 1:numel(mask)
                        if mask(idx) <= dropoutRate
                            mask(idx) = 1;
                        else
                            mask(idx) = 0;
                        end
                    end
                    
                    a_next = mask.*a_next;
                    % (the end of drop-out)
                    
                    stack_a{1,layer} = a_next;
                    stack_z{1,layer} = z_next;
                    stack_mask{1,layer} = mask;
                    a_prev = a_next;
                end
                
                [a_next, z_next] = feedforward(a_prev, ...
                    weights{size(weights,2)}, biases{size(weights,2)}, 'softmax');
                stack_a{1,size(weights,2)} = a_next;
                stack_z{1,size(weights,2)} = z_next;
                
            otherwise
                for layer = 1:size(weights,2)-1
                    [a_next, z_next] = feedforward(a_prev, weights{layer}, biases{layer}, 'sigmoid');
                    
                    % Create a Masking matrix M for drop-out
                    mask = rand(size(a_next));
                    for idx = 1:numel(mask)
                        if mask(idx) <= dropoutRate
                            mask(idx) = 1;
                        else
                            mask(idx) = 0;
                        end
                    end
                    
                    a_next = mask.*a_next;
                    % (the end of drop-out)
                    
                    stack_mask{1,layer} = mask;
                    stack_a{1,layer} = a_next;
                    stack_z{1,layer} = z_next;
                    a_prev = a_next;
                end
                
                [a_next, z_next] = feedforward(a_prev, ...
                    weights{size(weights,2)}, biases{size(weights,2)}, 'sigmoid');
                stack_a{1,size(weights,2)} = a_next;
                stack_z{1,size(weights,2)} = z_next;
        end
        
%% ------------------------------------------------------------------------
    % If NO dropout setting
    case 'no'
        mask = 1; stack_mask = 1;
        a_prev = input;
        switch output_activationF
            case 'softmax'
                for layer = 1:size(weights,2)-1
                    [a_next, z_next] = feedforward(a_prev, ...
                        weights{layer}, biases{layer}, 'sigmoid');
                    stack_a{1,layer} = a_next;
                    stack_z{1,layer} = z_next;
                    a_prev = a_next;
                end
                
                [a_next, z_next] = feedforward(a_prev, ...
                    weights{size(weights,2)}, biases{size(weights,2)}, 'softmax');
                stack_a{1,size(weights,2)} = a_next;
                stack_z{1,size(weights,2)} = z_next;
                
            otherwise
                for layer = 1:size(weights,2)
                    [a_next, z_next] = feedforward(a_prev, weights{layer}, biases{layer}, 'sigmoid');
                    stack_a{1,layer} = a_next;
                    stack_z{1,layer} = z_next;
                    a_prev = a_next;
                end
        end
        
end
end
