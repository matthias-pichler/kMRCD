
    clc;
    clear all;
    close all;    
    rng(5);
    
    addpath(genpath(fileparts(which(mfilename))));
    
    FontSizeAxis = 15;
    FontSizeLabel = 12;
    FontSize = 10;

    color1 = [0    0.6980    0.9333];
    color2 = [0.9333    0.4627         0];
    
    %%%%    Set the example to run    
    runExample = 1;
    
    %%%%    Set the contamination degree
    epsilon = 0.2;
    
    %%%%    Set the expected amaount of regular obs.    
    alpha = 0.75;
    
    if runExample==2    
         N = 1000; N1 = ceil((1-epsilon)*N); N2 = N - N1;        
         data = halfkernel(N1,N2, -20, 20, 40, 5, 0.6);  
         mask = logical(data(:,3));
         x = data(:, 1:2);
         y = data(:, 3);   
         ind = randperm(size(x, 1), size(x, 1));
         x=x(ind, :);
         y=y(ind, :);
         x = rZscores(x);
         kModel = AutoRbfKernel(x); 
    else
        ndm = NewDataModel(ALYZCorrelationType(), ClusterContamination());
        [x, ~, ~,idxOutliers] = ndm.generateDataset(1000, 2, epsilon, 15);        
        y = ones(1000,1);
        y(idxOutliers) = 0;
        y = logical(y);
        x = rZscores(x);
        kModel = LinKernel(); 
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%
    %%%     Run the kMRCD algorithm... 
    %%%    
    
    poc = kMRCD(kModel); 
    solution = poc.runAlgorithm(x, alpha);  
    
    %   Returns:
    %   -   hsubsetIndices: indices of the h-subset elements.
    %   -   flaggedOutlierIndices:  indices of the h-flagged outliers.
    %   -   rd and cutoff    
    
    disp('We have the following solution:');
    disp(solution);
    
    rho = solution.rho;
    scfac = solution.scfac;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%
    %%%%    Visualisation    

    %mi = [ floor(min(x)); ymin = mi(1); xmin = mi(2);
    %ma = ceil(max(x)); ymax = ma(1);  xmax = ma(2);  
    [rr, cc] = meshgrid(-5:0.1:5, -5:0.1:5);        
    yy=[rr(:), cc(:)];    

    Kx = kModel.compute(x(solution.hsubsetIndices, :), x(solution.hsubsetIndices, :));
    nx = size(Kx,1);
    Kt = kModel.compute(yy, x(solution.hsubsetIndices, :)); 
    Kc = Utils.center(Kx);
    Kt_c = Utils.center(Kx,Kt);
    Ktt_diag = diag(kModel.compute(yy,yy)); % Precompute
    Kxx = Ktt_diag - (2/nx)*sum(Kt,2) + (1/nx^2)*sum(sum(Kx));
    smdMesh = (1/rho)*(Kxx - (1-rho)*scfac*sum((Kt_c/((1-rho)*scfac*Kc + nx*rho*eye(nx)).*Kt_c),2)); 

    ss = logical(y);

    fig = figure(1);         
    contour(rr, cc, reshape(log(smdMesh), size(rr)), 20); hold on;
    plot(x(ss, 1), x(ss, 2), '.', 'Color', color1, 'MarkerSize', 12);
    plot(x(~ss, 1), x(~ss, 2), '.', 'Color', color2, 'MarkerSize', 12);
    set(gca,'FontSize',FontSize);
    %plot(x(5000:6000, 1), x(5000:6000, 2), '.m', 'MarkerSize', 20);
    colormap bone;
    set(gcf,'color','w');
    ylim([-4, 4]);
    title('Input dataset with marked outliers');


    fig = figure(2);         
    contour(rr, cc, reshape(log(smdMesh), size(rr)), 20); hold on;        
    plot(x(:, 1), x(:, 2), '.', 'Color', color2, 'MarkerSize', 12);
    plot(x(solution.hsubsetIndices, 1), x(solution.hsubsetIndices, 2), '.', 'Color', color1, 'MarkerSize', 12);        
    set(gca,'FontSize',FontSize);
    hold off;
    colormap bone;
    set(gcf,'color','w');
    ylim([-4, 4]);
    title('h-subset');


    fig = figure(3);        
    contour(rr, cc, reshape(log(smdMesh), size(rr)), 20); hold on;
    plot(x(:, 1), x(:, 2), '.', 'Color', color1, 'MarkerSize', 12);
    plot(x(solution.flaggedOutlierIndices, 1), x(solution.flaggedOutlierIndices, 2), '.', 'Color', color2, 'MarkerSize', 12);               
    hold off;
    colormap bone;
    set(gcf,'color','w');
    ylim([-4, 4]);
    set(gca,'FontSize',FontSize);
    title('Flagged outliers');

    fig=  figure(4);         
    scatter(1:length(solution.rd),solution.rd,[],color2,'filled'); hold on;
    %scatter(x(solution.hsubsetIndices, :), rd(solution.hsubsetIndices),[],color1,'filled'); 
    plot(1:length(solution.rd),repmat(solution.cutoff,length(solution.rd),1),'k','LineWidth',2)   
    set(gca,'FontSize',FontSize);
    hold off;
    title('Robust distances and flagging threshold');
    
    
    
    
    
    
    
