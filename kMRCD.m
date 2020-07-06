classdef kMRCD < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = private)        
        kModel;                         %   Used kernel model        
        cStepIterationsAllowed = 100;   %   Maximum number of CStep iterations allowed        
        maxcond = 50;                   %   Condition number one wants to achieve
    end
       
    methods (Access = public)
        
        function this = kMRCD(kModel)
            if ~isempty(kModel)
                this.kModel = kModel;
            else
                this.kModel = LinKernel();
            end
        end
 
         function solution = runAlgorithm(this, x, alpha)                        
            assert(alpha<=1 && alpha>=0.5, 'The percentage of regular observations, alpha, should be in [0.5-1]'); 
            K = this.kModel.compute(x, x);   
            [n, p] = size(x);
            
            %   Grab observation ranking from initial estimators
            solution = struct();            
            solution(1).outlyingnessIndices = Utils.SDO(K,alpha);
            solution(1).name = 'SDO';                
            solution(2).outlyingnessIndices = Utils.SpatialRank(K,alpha);  
            solution(2).name = 'SpatialRank';              
            solution(3).outlyingnessIndices = Utils.SpatialMedianEstimator(K,alpha);
            solution(3).name = 'SpatialMedian';
            solution(4).outlyingnessIndices = Utils.SSCM(K);
            solution(4).name = 'SSCM';
    
            scfac = Utils.MCDcons(p, alpha) ;
            
            %   For all initial estimators, do:
            rhoL = NaN(numel(solution),1);
            for index = 1:numel(solution)                
                % Compute solution for each estimator; 
                solution(index).hsubsetIndices = solution(index).outlyingnessIndices(1:ceil(n*alpha));                
                % Determine rho for each estimator            
                s = svd(Utils.center(this.kModel.compute(x(solution(index).hsubsetIndices, :), x(solution(index).hsubsetIndices, :))));
                nx = length(solution(index).hsubsetIndices);                 
                e_min = min(s); 
                e_max = max(s);
                fncond = @(rho) (nx*rho + (1-rho)*scfac*e_max)/(nx*rho + (1-rho)*scfac*e_min) - this.maxcond;
                try
                    rhoL(index) = fzero(fncond, [10^(-6),0.99]);
                catch 
                    % Find value closest to maxcond instead
                    disp('Running grid search');
                    grid = linspace(0.000001,1-0.000001,1000);
                    objgrid = abs(arrayfun(fncond,grid));
                    rhoL(index) = min(grid(objgrid == min(objgrid)));
                end                
            end 
            
            % Set rho as max of the rho_i's obtained for each subset in previous step                        
            rho = max(rhoL(rhoL <= max([0.1, median(rhoL)])));
            
            Ktt_diag = diag(K); 
            for index=1:numel(solution)                              
               for iteration = 1:this.cStepIterationsAllowed        
                    hSubset = solution(index).hsubsetIndices;    
                    Kx = this.kModel.compute(x(hSubset, :), x(hSubset, :));
                    nx = size(Kx,1);
                    Kt = this.kModel.compute(x, x(hSubset, :)); 
                    Kc = Utils.center(Kx);
                    Kt_c = Utils.center(Kx,Kt);
                    Kxx = Ktt_diag - (2/nx)*sum(Kt,2) + (1/nx^2)*sum(sum(Kx));
                    smd = (1/rho)*(Kxx - (1-rho)*scfac*sum((Kt_c/((1-rho)*scfac*Kc + nx*rho*eye(nx)).*Kt_c),2));                     
                    [~, indices] = sort(smd);                                       
                    
                    % Redefine the h-subset
                    solution(index).hsubsetIndices = indices(1:nx);                       
                    if (setdiff(hSubset, solution(index).hsubsetIndices))
                        disp(['Convergence at iteration ' mat2str(iteration) ', ' solution(index).name]);                                                
                        sigma = svd(Kc);
                        sigma = (1-rho)*scfac*sigma + numel(solution(index).hsubsetIndices)*rho;                        
                        solution(index).obj =  sum(log(sigma));
                        solution(index).smd = smd;
                        break;                    
                    end                                        
               end
               assert(iteration<this.cStepIterationsAllowed, 'no c-step convergence');
            end

            %   Select the solution with the lowest objective function            
            [~, mIndex] = min([solution.obj]);     
            %   ...and remove the other solutions
            solution = solution(mIndex);                             
            disp(['-> Best estimator is ' solution.name]);
            
            % Determine cut-off for outlier flagging
            solution.rd = max(sqrt(solution.smd),0);
            solution.rho = rho;
            solution.scfac = scfac;
            
            %   Flag outliers
            solution.ld = log(0.1 + solution.rd);                                    
            [tmcd,smcd] = unimcd(solution.ld, numel(solution.hsubsetIndices));
            solution.cutoff = exp(tmcd + norminv(0.995) * smcd) - 0.1;
            solution.flaggedOutlierIndices = find(solution.rd > solution.cutoff);         
        end
        
    end
    
end

