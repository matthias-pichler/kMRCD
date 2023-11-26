%% Shakespear vs Trump
% Compute the case study of shakespear vs trump

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

projectDir = fileparts(fileparts(which(mfilename)));

imageDir = fullfile(projectDir, 'images', 'shakespear_trump');
tableDir = fullfile(projectDir, 'tables', 'shakespear_trump');
datasetDir = fullfile(projectDir, 'datasets', 'shakespear_trump');

modelName = "all-mpnet-base-v2";

file = fullfile(datasetDir, ['shakespear_trump' '_' char(modelName) '.parquet']);

mkdir(imageDir);
mkdir(tableDir);

%% Load Data

N = 1000;
[eps20, eps20Labels] = generateSample(file, N, 0.2);
data = eps20;
labels = eps20Labels;

%% e=0.2, a=0.7

alpha = 0.7;

Y = tsne(data, Distance="cosine");
fig = figure(1);
textscatter(Y,string(labels),ColorData=labels,TextDensityPercentage=0);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "eps20_tsne.png"),'png');

clear Y;

kModel = AutoSphereRbfKernel(data);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(data, alpha);

% h Subset
hSubset = table(labels(solution.hsubsetIndices), VariableNames="label");
hSubsetSummary = groupcounts(hSubset, "label");
writetable(hSubsetSummary, fullfile(tableDir, "eps20_h_subset.csv"));

clear hSubset hSubsetSummary;

% Confusion Matrix
grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

cm = confusionmat(labels,grouphat);

fig = figure(2);
confusionchart(fig, cm, categories(labels));
saveas(fig, fullfile(imageDir, 'eps20_confusion_matrix.png'),'png');

clear cm grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'eps20_mahalanobis_distances.png'),'png');

% Comparison
fig = figure(4);
stats = evaluation(data, labels, alpha, solution);
saveas(fig, fullfile(imageDir, 'eps20_pr_curve.png'),'png');
writetable(stats, fullfile(tableDir, "eps20_comparison.csv"));

clear stats;

clear solution kModel alpha poc;
clear data labels;
%% Comparison

[eps0, eps0Labels] = generateSample(file, N, 0);
[eps40, eps40Labels] = generateSample(file, N, 0.4);

stats = [
        runComparison(eps0, eps0Labels, 0, 0.5); runComparison(eps0, eps0Labels, 0, 0.75); runComparison(eps0, eps0Labels, 0, 0.9); ...
        runComparison(eps20, eps20Labels, 0.2, 0.5); runComparison(eps20, eps20Labels, 0.2, 0.75); runComparison(eps20, eps20Labels, 0.2, 0.9); ...
        runComparison(eps40, eps40Labels, 0.4, 0.5); runComparison(eps40, eps40Labels, 0.4, 0.75); runComparison(eps40, eps40Labels, 0.4, 0.9)
        ];

writetable(stats, fullfile(tableDir, ['comparison' '_' char(modelName) '.csv']));

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

function stats = runComparison(data, labels, outlierContamination, robustness)
    arguments
        data double {mustBeNonempty}
        labels categorical {mustBeNonempty}
        outlierContamination (1,1) double {mustBeInRange(outlierContamination,0,1)}
        robustness (1,1) double {mustBeInRange(robustness,0.5,1)}
    end
    
    kModel = AutoSphereRbfKernel(data);
    poc = kMRCD(kModel); 
    solution = poc.runAlgorithm(data, robustness);

    stats = evaluation(data, labels, robustness, solution);

    names = arrayfun(@(s) sprintf('%s (e=%0.2f, a=%0.2f)', s, outlierContamination, robustness), string(stats.Properties.RowNames), UniformOutput=false);

    stats = [table(names, VariableNames="Name"), stats];
    stats.Properties.RowNames = names;
end