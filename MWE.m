
    clc;
    close all;
    clear all;
    rng('default');
    
    %%% Include the code of all subfolders
    addpath(genpath(fileparts(which(mfilename)))); 
    %%% Nbr. of observations in the dataset [1, ..., 1000]
    n = 500;    
    %%% Contamination amount in the dataset [0, ..., 0.5]
    eps = 0.20;    
    %%% Expected % regular observations [0.5, ..., 1]
    alpha = 0.70;    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%
    %%%%%   First example: normal distributed data with cluster
    %%%%%   contamination and linear kernel
    %%%%%
    
    %%% Generate the dataset x with label vector y 
    x = [mvnrnd([2, 3], [1 1.5; 1.5 3], n); mvnrnd([-3, 3], 0.1*eye(2,2), floor(n*eps))];
    y = [+1 * ones(n,1); -1 * ones(floor(n*eps),1)];    
    %%% Perform robust standarization
    x = rZscores(x);
    
    %%% Create an estimator instance
    kmrcd = kMRCD(LinKernel());    
    %%% Run kMRCD algorithm
    solution = kmrcd.runAlgorithm(x, alpha);
    
    %%% Display the solution    
    disp('kMRCD solution is:');
    disp(solution)
    
    figure; 
    subplot(1,3,1);
    plot(x(y>0, 1), x(y>0, 2), '.g'); hold on;
    plot(x(y<0, 1), x(y<0, 2), '.r');
    title('Dataset with outliers');

    subplot(1,3,2);
    plot(x(:, 1), x(:, 2), '.r'); hold on;
    plot(x(solution.hsubsetIndices, 1), x(solution.hsubsetIndices, 2), '.g');     
    title(['kMRCD cstepmask for h = ' mat2str(alpha) ' n']);
    
    subplot(1,3,3);
    plot(x(:, 1), x(:, 2), '.g'); hold on;
    plot(x(solution.flaggedOutlierIndices, 1), x(solution.flaggedOutlierIndices, 2), '.r');
    title(['kMRCD Flagged outliers for h = ' mat2str(alpha) ' n']);
    drawnow();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%
    %%%%%   Second example: non-linear data with half moon 
    %%%%%   contamination and rbf kernel with bandwidth estimation
    %%%%%
    
    x = halfkernel(n,floor(n*eps), -20, 20, 40, 5, 0.6);      
    y = x(:, 3);          
    x = x(:, 1:2);
    
    %%% Perform robust standarization
    x = rZscores(x);
    
    %%% Create an estimator instance with rbf kernel
    kmrcd = kMRCD(AutoRbfKernel(x));    
    %%% Run kMRCD algorithm
    solution = kmrcd.runAlgorithm(x, alpha);    
    
    %%% Display the solution     
    disp('kMRCD solution is:');
    disp(solution)
    
    figure; 
    subplot(1,3,1);
    plot(x(y>0, 1), x(y>0, 2), '.g'); hold on;
    plot(x(y<0, 1), x(y<0, 2), '.r');
    title('Dataset with outliers');

    subplot(1,3,2);
    plot(x(:, 1), x(:, 2), '.r'); hold on;
    plot(x(solution.hsubsetIndices, 1), x(solution.hsubsetIndices, 2), '.g');     
    title(['kMRCD cstepmask for h = ' mat2str(alpha) ' n']);
    
    subplot(1,3,3);
    plot(x(:, 1), x(:, 2), '.g'); hold on;
    plot(x(solution.flaggedOutlierIndices, 1), x(solution.flaggedOutlierIndices, 2), '.r');
    title(['kMRCD Flagged outliers for h = ' mat2str(alpha) ' n']);
    drawnow();
    
    disp('That is all, folks!');
    
    
    
    