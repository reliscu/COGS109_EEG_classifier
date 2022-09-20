% NEXT STEPS:
% 1) CREATE ROC PLOT FOR EACH MODEL 
% 2) use other features, e.g. skewness and kurtosis 
 
% categories
num_subs = 14;
negative = zeros(num_subs, 1); % healthy subs
positive = ones(num_subs, 1); % SZ subs
categories = [negative; positive];
 
% predictors
predictors = [e1_var e2_var e3_var e4_var e5_var e6_var e7_var e8_var ...
    e9_var e10_var e11_var e12_var e13_var e14_var e15_var e16_var ...
    e17_var e18_var e19_var];
 
n_models = size(predictors, 2); % number of predictors = number of models
n_subs = size(predictors, 1); % number of observations = number of subjects
thresholds = 0.1:0.05:.95;
% thresholds = [0.3 0.5 0.7]; for testing
mse_per_threshold_per_model = [];
 
tic
for t = 1:length(thresholds) % test varying thresholds for classifier
 
best_predictors = [];
used_predictors = [];
predictor_inds_avail = 1:n_models; % predictors available as forward selection progresses
ccr = [];
ccr_per_model = [];
 
for p = 1:n_models
    
    % for each model with p predictors: try out each predictor
    for j = 1:n_models
       % check that predictor is not already in model
        if ~isnan(predictor_inds_avail(j)) == 1
            predictors_test = [best_predictors predictors(:, j)];
        
        % find training and test mse:
            % randomize subjects for k-fold cross-validation
            data_rdm = [predictors_test categories];
            data_rdm = data_rdm(randperm(n_subs), :);
            predictors_rdm = data_rdm(:, 1:end - 1);
            categories_rdm = data_rdm(:, end);
            interval_inds = 1:(n_subs + 1);
            
            for k = 1:n_subs % cross-val at level of subjects
                test_inds = false(n_subs, 1); 
                % select subject for test set:
                test_inds(k) = true;
                % assign remaining subjects to training set:
                train_inds = ~test_inds;
 
                % fit model using jth electrode training obs:
                mdl = fitglm(predictors_rdm(train_inds, :), ...
                    categories(train_inds), 'distribution', 'binomial');
                yhat = predict(mdl, predictors_rdm);
                yhat = yhat > 0.5; % thresholds(t); % threshold changes each pass
                ccr_train_temp(:, k) = mean(categories(train_inds) == ...
                    yhat(train_inds));
                ccr_test_temp(k) = categories(test_inds) == ...
                    yhat(test_inds);
            end % end 'k'
            ccr_train(j) = mean(ccr_train_temp);
            ccr_test(j) = mean(ccr_test_temp);
        else 
            ccr_train(j) = NaN;
            ccr_test(j) = NaN;
        end % end 'if'   
    end % end 'j' 
     
    % store test MSE per model
    [ccr_max_test, idx_max_test] = max(ccr_test);
    ccr_per_model_test(p) = ccr_max_test;
    
    % store train MSE per model
    [ccr_max_train, idx_max_train] = max(ccr_train);
    ccr_per_model_train(p) = ccr_max_train;
    
    % best predictor within each model complexity is a function ...
    % ... of train MSE
    predictor_inds_avail(idx_max_train) = NaN;
    % save best predictor
    best_predictors = [best_predictors predictors(:, idx_max_train)];
    % note: each predictor^ is chosen based on highest train CCR, since ...
    % ... model complexity is comparable 
    
end % end 'p'
 
ccr_per_model_per_threshold(:, t) = ccr_per_model_test';
% each model's test CCR is recorded for ultimate model selection
 
end % end 't'
time = toc/60;
 
% plot number of predictors vs. correct classification rate
num_pred = 1:length(ccr_per_model_test);
figure(1)
hold on
plot(num_pred, ccr_per_model_test, 'o-')
plot(num_pred, ccr_per_model_train, 'o-')
xlabel('Number of Predictors')
ylabel('Correct Classification Rate')
title('Number of Predictors vs. Correct Classification Rate') 
legend('Test MSE', 'Train MSE', 'location', 'northwest')
 
% plot number of predictors vs. CCR for each threshold
color = jet(length(thresholds));
figure(2)
for i = 1:length(thresholds)
    plot(num_pred, ccr_per_model_per_threshold(:, i), 'marker', 'o', ...
        'color', color(i, :));
    hold on
end 
title('Number of Predictors vs. Correct Classification Rate')
xlabel('Number of Predictors')
ylabel('Correct Classification Rate')
legend('Threshold = 0.1', 'Threshold = 0.15', 'Threshold = 0.2', ...
'Threshold = 0.25', 'Threshold = 0.3', 'Threshold = 0.35', ...
   'Threshold = 0.4', 'Threshold = 0.45', 'Threshold = 0.5', ...
   'Threshold = 0.55', 'Threshold = 0.6', 'Threshold = 0.65', ...
   'Threshold = 0.7', 'Threshold = 0.75', 'Threshold = 0.8', ...
   'Threshold = 0.85', 'Threshold = 0.9', 'Threshold = 0.95', ...
   'location', 'bestoutside')
hold off 
 
% plot ROC curves for each "best" model:
figure(2)
for j = 1:n_models % for each "best" model ...
    for i = 1:length(thresholds) % ... find TP and FP rates per threshold
        mdl_best = fitglm(best_predictors(:, 1:j), categories);
        yhat_best_model = predict(mdl_best) > thresholds(i);
        yhat_best_model = double(yhat_best_model);
 
        conf_matrix = confusionmat(yhat_best_model, categories);
        tp_rate(i, j) = conf_matrix(2, 2)/sum(conf_matrix(2, :));
        fp_rate(i, j) = conf_matrix(1, 2)/sum(conf_matrix(1, :));
    end 
    
    plot(fp_rate(:, j), tp_rate(:, j))
    hold on
end 
title('ROC Curve per Model')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
legend('1 predictor', '2 predictors', '3 predictors', '5 predictors', ...
    '6 predictors', '7 predictors', '8 predictors', '9 predictors', ...
    '10 predictors', '11 predictors', '12 predictors', '13 predictors', ...
    '14 predictors', '15 predictors', '16 predictors', '17 predictors', ...
    '18 predictors', '19 predictors', 'location', 'bestoutside')
hold off
