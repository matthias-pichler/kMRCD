%% Shakespear vs Trump
% Compute the case study of shakespear vs trump

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

mkdir ../images shakespear_trump
mkdir ../tables shakespear_trump

%% Load Data

N = 1000;
[data,labels] = generateSample(N, 0.2);

%% e=0.2, a=0.7

alpha = 0.7;

Y = tsne(data, Distance="cosine");
fig = figure(1);
textscatter(Y,string(labels),ColorData=labels,TextDensityPercentage=0);
title("t-SNE Embeddings");
set(fig,'color','w');
saveas(fig,'../images/shakespear_trump/eps20_tsne.png','png');

clear Y;

kModel = AutoSphereRbfKernel(data);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(data, alpha);

% h Subset
hSubset = table(labels(solution.hsubsetIndices), VariableNames="label");
hSubsetSummary = groupcounts(hSubset, "label");
writetable(hSubsetSummary, "../tables/shakespear_trump/eps20_h_subset.csv");

clear hSubset hSubsetSummary;

% Confusion Matrix
grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

cm = confusionmat(labels,grouphat);

fig = figure(2);
confusionchart(fig, cm, categories(labels));
saveas(fig,'../images/shakespear_trump/eps20_confusion_matrix.png','png');

clear cm grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig,'../images/shakespear_trump/eps20_mahalanobis_distances.png','png');

% Comparison
fig = figure(4);
stats = evaluation(data, labels, alpha, solution);
saveas(fig,'../images/shakespear_trump/eps20_pr_curve.png','png');
writetable(stats, "../tables/shakespear_trump/eps20_comparison.csv");

clear stats;

clear solution kModel alpha poc;
%% Comparison

[eps0, eps0Labels] = generateSample(N, 0);
[eps20, eps20Labels] = generateSample(N, 0.2);
[eps40, eps40Labels] = generateSample(N, 0.4);

stats = [
        runComparison(eps0, eps0Labels, 0, 0.5); runComparison(eps0, eps0Labels, 0, 0.75); runComparison(eps0, eps0Labels, 0, 0.9); ...
        runComparison(eps20, eps20Labels, 0.2, 0.5); runComparison(eps20, eps20Labels, 0.2, 0.75); runComparison(eps20, eps20Labels, 0.2, 0.9); ...
        runComparison(eps40, eps40Labels, 0.4, 0.5); runComparison(eps40, eps40Labels, 0.4, 0.75); runComparison(eps40, eps40Labels, 0.4, 0.9)
        ];

writetable(stats, "../tables/shakespear_trump/comparison.csv");

%% Functions

function [embeddings,labels] = generateSample(sampleSize, outlierContamination)
    arguments
        sampleSize (1,1) double {mustBeInteger, mustBePositive}
        outlierContamination (1,1) double {mustBeInRange(outlierContamination,0,1)}
    end

    rawData = parquetread("../datasets/shakespear_trump/shakespear_trump_all-mpnet-base-v2.parquet", SelectedVariableNames=["text", "author", "embedding"]);
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