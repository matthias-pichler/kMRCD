%% Shakespear vs Trump
% Compute the case study of shakespear vs trump

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

modelName = 'all-mpnet-base-v2';
% modelName = 'bge-large-en-v1.5';
% modelName = 'bge-small-en-v1.5';
% modelName = 'all-MiniLM-L6-v2';

datasetName = 'shakespear_trump';
% datasetName = 'shakespear_trump_cleaned';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

mkdir(imageDir, modelName);
mkdir(tableDir, modelName);

imageDir = fullfile(imageDir, modelName);
tableDir = fullfile(tableDir, modelName);

%% Visualize

[embeddings, labels] = generateSample(file, 1000, 0.2);

Y = tsne(embeddings);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Sample

alpha = 0.7;
N = 1000;

kModel = AutoSphereRbfKernel(embeddings);
% kModel = AutoRbfKernel(data);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(embeddings, alpha);

% h Subset
hSubset = table(labels(solution.hsubsetIndices), VariableNames="label");
hSubsetSummary = groupcounts(hSubset, "label");
writetable(hSubsetSummary, fullfile(tableDir, modelName, "e02_h_subset.csv"));

clear hSubset hSubsetSummary;

% Confusion Matrix
grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

cm = confusionmat(labels,grouphat);

fig = figure(2);
confusionchart(fig, cm, categories(labels));
saveas(fig, fullfile(imageDir, modelName, 'confusion_matrix.png'),'png');

clear cm grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, modelName, 'mahalanobis_distances.png'),'png');

% Comparison
fig = figure(4);
stats = evaluation(embeddings, labels, alpha, solution, Estimators={'lof' 'iforest'});
saveas(fig, fullfile(imageDir, modelName, 'pr_curve.png'),'png');

clear stats;
clear solution kModel alpha poc;
clear embeddings labels;

set(0,'DefaultFigureVisible','off');

%% Run a = 0.5, e = 0.1

stats = runComparison(iter=100, alpha=0.5, data=@()generateSample(file, N, 0.1));

writetable(stats, fullfile(tableDir, "comparison_a05_e01.csv"));

%% Run a = 0.75, e = 0.1

stats = runComparison(iter=100, alpha=0.75, data=@()generateSample(file, N, 0.1));

writetable(stats, fullfile(tableDir, "comparison_a075_e01.csv"));

%% Run a = 0.9, e = 0.1

stats = runComparison(iter=100, alpha=0.9, data=@()generateSample(file, N, 0.1));

writetable(stats, fullfile(tableDir, "comparison_a09_e01.csv"));

%% Run a = 0.5, e = 0.2

stats = runComparison(iter=100, alpha=0.5, data=@()generateSample(file, N, 0.2));

writetable(stats, fullfile(tableDir, "comparison_a05_e02.csv"));

%% Run a = 0.75, e = 0.2

stats = runComparison(iter=100, alpha=0.75, data=@()generateSample(file, N, 0.2));

writetable(stats, fullfile(tableDir, "comparison_a075_e02.csv"));

%% Run a = 0.5, e = 0.3

stats = runComparison(iter=100, alpha=0.5, data=@()generateSample(file, N, 0.3));

writetable(stats, fullfile(tableDir, "comparison_a05_e03.csv"));

%% Functions

function [embeddings,labels] = generateSample(filepath, sampleSize, outlierContamination)
    arguments
        filepath (1,1) string {mustBeFile}
        sampleSize (1,1) double {mustBeInteger, mustBePositive}
        outlierContamination (1,1) double {mustBeInRange(outlierContamination,0,1)}
    end

    rawData = parquetread(filepath, SelectedVariableNames=["text", "author", "embedding"]);
    rawData.author = categorical(rawData.author);

    shakespearIndices = find(rawData.author == "shakespear");
    trumpIndices = find(rawData.author == "trump");
    
    numShakespearSamples = round(outlierContamination * sampleSize);
    numTrumpSamples = sampleSize - numShakespearSamples;
    
    shakespearSampleIndices = datasample(shakespearIndices, numShakespearSamples, Replace=false);
    trumpSampleIndices = datasample(trumpIndices, numTrumpSamples, Replace=false);
    data = rawData(sort(cat(1, shakespearSampleIndices, trumpSampleIndices)),:);

    embeddings = cell2mat(cellfun(@transpose,data.embedding, UniformOutput=false));
    labels = renamecats(data.author, {'trump' 'shakespear'}, {'inlier' 'outlier'});

    perm = randperm(height(embeddings));
    embeddings = embeddings(perm, :);
    labels = labels(perm, :);
end

function stats = runComparison(NameValueArgs)
    arguments
        NameValueArgs.iter (1,1) double {mustBeInteger, mustBePositive} = 100
        NameValueArgs.alpha (1,1) double {mustBeInRange(NameValueArgs.alpha, 0.5, 1)}
        NameValueArgs.data
    end

    alpha = NameValueArgs.alpha;
    iter = NameValueArgs.iter;

    kMRCDStats = table(Size=[iter, 6], ...
        VariableNames={'accuracy' 'precision' 'sensitivity' 'specificity' 'f1Score' 'aucpr'}, ...
        VariableTypes=repmat("double", 1, 6));
    lofStats = kMRCDStats;
    iforestStats = kMRCDStats;

    for i = 1:iter
        fprintf("Iteration: %d\n", i);

        if(isa(NameValueArgs.data, 'function_handle'))
            [data, labels] = NameValueArgs.data();
        end
        
        kModel = K1Kernel(data);
        
        poc = kMRCD(kModel); 
        solution = poc.runAlgorithm(data, alpha);
    
        e = evaluation(data, labels, alpha, solution, Estimators={'lof' 'iforest'});
    
        kMRCDStats(i,:) = e('kMRCD',:);
        lofStats(i,:) = e('lof', :);
        iforestStats(i,:) = e('iforest', :);
    end
    
    stats = vertcat(harmmean(kMRCDStats), harmmean(lofStats), harmmean(iforestStats));
    stats = horzcat(table(["kMRCD";"lof";"iforest"], VariableNames={'name'}),stats);
    stats.Properties.RowNames = stats.name;
end