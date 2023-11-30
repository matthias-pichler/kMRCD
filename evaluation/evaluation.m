function result = evaluation(data, labels, alpha, solution, NameValueArgs)
    arguments
        data (:,:) double
        labels (:,1) categorical
        alpha (1,1) double {mustBeInRange(alpha,0.5,1)}
        solution struct
        NameValueArgs.CategoricalPredictors = []
        NameValueArgs.Estimators {mustBeMember(NameValueArgs.Estimators, {'lof' 'iforest' 'rrcforest' 'ocsvm' 'robustcov'})} = {'lof' 'iforest' 'robustcov'}
    end

    assert(isequal(size(labels), [height(data), 1]))
    
    stats = struct();
    scores = struct();

    % kMRCD
    grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
    grouphat(solution.flaggedOutlierIndices) = "outlier";
    stats.kMRCD = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
    scores.kMRCD = solution.rd;

    % MCD
    if ismember('robustcov', NameValueArgs.Estimators)
        tic;
        [~, ~, sc, tf] = robustcov(data, OutlierFraction=(1-alpha));
        t = toc;
        fprintf("robustcov: %0.4f sec\n", t);

        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(tf) = "outlier";
        stats.robustcov = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
        scores.robustcov = sc;
    end
    
    % Local Outlier Factor
    if ismember('lof', NameValueArgs.Estimators)
        tic;
        [~,tf, sc] = lof(data, ContaminationFraction=(1-alpha), ...
            CategoricalPredictors=NameValueArgs.CategoricalPredictors);
        t = toc;
        fprintf("lof: %0.4f sec\n", t);

        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(tf) = "outlier";
        stats.lof = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
        scores.lof = sc;
    end

    % Isolation Forest
    if ismember('iforest', NameValueArgs.Estimators)
        tic;
        [~,tf, sc] = iforest(data, ContaminationFraction=(1-alpha), ...
            CategoricalPredictors=NameValueArgs.CategoricalPredictors);
        t = toc;
        fprintf("iforest: %0.4f sec\n", t);
        
        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(tf) = "outlier";
        stats.iforest = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
        scores.iforest = sc;
    end

    % Random Robust Cut Forest
    if ismember('rrcforest', NameValueArgs.Estimators)
        tic;
        [~,tf,sc] = rrcforest(data, ContaminationFraction=(1-alpha), ...
            CategoricalPredictors=NameValueArgs.CategoricalPredictors);
        t = toc;
        fprintf("rrcforest: %0.4f sec\n", t);
        
        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(tf) = "outlier";
        stats.rrcforest = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
        scores.rrcforest = sc;
    end

    % One-Class Support Vector Machine
    if ismember('ocsvm', NameValueArgs.Estimators)
        tic;
        [~,tf,sc] = ocsvm(data, ContaminationFraction=(1-alpha), ...
            CategoricalPredictors=NameValueArgs.CategoricalPredictors);
        t = toc;
        fprintf("ocsvm: %0.4f sec\n", t);
        
        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(tf) = "outlier";
        stats.ocsvm = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
        scores.ocsvm = sc;
    end

    clear tf;
    clear sc;

    % only print prcurves for contaminated datasets
    if numel(unique(labels)) == 2
        outlierRatio = sum(labels == "outlier")/numel(labels);
    
        hold on;
    
        xlabel("Recall");
        ylabel("Precision");
        title("Precision-Recall Curve");
    
        fn = fieldnames(scores);
        colors = jet(numel(fn));
        for k=1:numel(fn)
            s = scores.(fn{k});
            auc = prcurve(labels,s,'outlier',DisplayName=fn{k}, Color=colors(k,:));
            stats.(fn{k}).aucpr = auc;
        end
        yline(outlierRatio, LineStyle="--", ...
            DisplayName=sprintf("No Skill Classifier (AUC=%0.4f)", outlierRatio));
        legend;
        hold off;
    else
        fn = fieldnames(scores);
        for k=1:numel(fn)
            stats.(fn{k}).aucpr = NaN;
        end
    end

    s = stats.kMRCD;
    
    fn = fieldnames(stats);
    for i=2:numel(fn)
        s = vertcat(s, stats.(fn{i}));
    end

    result = struct2table(s, RowNames=fieldnames(stats));
end

