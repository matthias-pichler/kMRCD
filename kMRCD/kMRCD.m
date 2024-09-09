classdef kMRCD < handle
    %
    %kMRCD: Outlier detection in non-elliptical data by kernel MRCD, J.
    %       Schreurs, I. Vranckx et al.
    %
    %   The minimum regularized covariance determinant (MRCD) is a robust
    %   estimator for multivariate location and scatter, which detects outliers by fitting a
    %   robust covariance matrix to the data. The MRCD assumes that the observations
    %   are elliptically distributed. However, this property does not always apply to modern datasets.
    %   Together with the time criticality of industrial processing, small n,
    %   large p problems pose a challenging problem for any data analytics procedure. Both
    %   shortcomings are solved with the proposed kernel Minimum Regularized Covariance
    %   Determinant estimator, where we exploit the kernel trick to speed-up computations.
    %   More specifically, the MRCD location and scatter matrix estimate are computed in
    %   a kernel induced feature space, where regularization ensures that the covariance matrix is well-conditioned,
    %   invertible and defined for any dataset dimension.
    %
    %   Required input argument:
    %   x : a vector or matrix whose columns represent variables, and rows represent observations.
    %        Missing and infinite values are not allowed and should be excluded from the computations by the user.
    %   alpha : (1-alpha) measures the fraction of outliers the algorithm should
    %          resist. Any value between [0.5 <= alpha <= 1] may be specified. (default = 0.75)
    %   kModel : the kernel transformation used.
    %
    %   Output :  The output structure 'solution' contains the final results, with the following fields :
    %       -   solution.outlyingnessIndices: outlyingness weight of each
    %       observation according.
    %       -   solution.name: the name of the best initial estimator which was used to construct the final solution.
    %       -   solution.hsubsetIndices: the h-subset element indices after
    %       C-step convergence.
    %       -   solution.obj: the MRCD objective value.
    %       -   solution.smd: the Squared Mahanobis Distance values of each
    %       observation
    %       -   solution.rho: regularisation factor used
    %       -   solution.scfac: finite sample correction factor used
    %       -   solution.rd: Mahanobis distance value of each observation.
    %       -   solution.ld: rescaled Mahanobis distance values, defined as log(0.1 + solution.rd);
    %       -   solution.cutoff: outlier flagging cut-off
    %       -   solution.flaggedOutlierIndices: indices of the flagged
    %       outliers
    %
    %   Minimal working example :
    %       -   Create an estimator instance for a linear kernel:
    %           kmrcd = kMRCD(LinKernel());
    %       -   Run the kMRCD algorithm with alpha = 0.75
    %           solution = kmrcd.runAlgorithm(x, alpha);
    %
    %   Last modified by Iwein Vranckx, 6/07/2020,
    %   https://be.linkedin.com/in/ivranckx
    %   Git repository: https://github.com/ivranckx/kMRCD.git
    %   Licenced under the Non-Profit Open Software License version 3.0 (NPOSL-3.0)
    %
    %   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    %   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    %   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    %   FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    %   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
    %   THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
    
    properties (Access = private)
        kModel;                         %   Used kernel model
        cStepIterationsAllowed (1,1) double {mustBePositive, mustBeInteger} = 100;   %   Maximum number of CStep iterations allowed
        maxcond (1,1) double = 50;                   %   Condition number one wants to achieve
        estimators {mustBeMember(estimators, {'SDO' 'SpatialRank' 'SpatialMedian' 'SSCM'})} = {'SDO' 'SpatialMedian' 'SSCM'};
        cutoffEstimator (1,1) string {mustBeMember(cutoffEstimator, {'lognormal' 'chisquare' 'skewedbox'})} = 'lognormal';
    end
    
    methods (Access = private)
        
        function cutoff = lognormalCutoff(this, robustDistances, h)
            arguments
                this kMRCD
                robustDistances (:,1) double
                h (1,1) double {mustBePositive, mustBeInteger}
            end
            
            logDistances = log(0.1 + robustDistances);
            
            [tmcd,smcd] = LIBRA.unimcd(logDistances, h);
            
            cutoff = exp(tmcd + norminv(0.995) * smcd) - 0.1;
        end

        function cutoff = chisquareCutoff(this, robustDistances, h)
            arguments
                this kMRCD
                robustDistances (:,1) double
                h (1,1) double {mustBePositive, mustBeInteger}
            end
            
            logDistances = log(0.1 + robustDistances);
            
            [tmcd,smcd] = LIBRA.unimcd(logDistances, h);
            
            cutoff = exp(tmcd + norminv(0.995) * smcd) - 0.1;
        end
        
        function cutoff = skewedboxCutoff(this, robustDistances, NameValueArgs)
            arguments
                this kMRCD
                robustDistances (:,1) double
                NameValueArgs.a (1,1) double {mustBeNegative} = -4
                NameValueArgs.b (1,1) double {mustBePositive} = 3
                NameValueArgs.whisker (1,1) double {mustBePositive} = 1.5
            end
            
            logDistances = log(0.1 + robustDistances);
            
            % define the median and the quantiles
            pctiles = prctile(logDistances,[25;75]);
            q1 = pctiles(1,:);
            q3 = pctiles(2,:);
            
            % find the extreme values (to determine where whiskers appear)
            medc = LIBRA.mc(logDistances);
            vhiadj = q3+NameValueArgs.whisker*exp(NameValueArgs.b*medc)*(q3-q1);  %Upper cutoff value for the adjusted boxplot.
            cutoff = max(logDistances(logDistances<=vhiadj));
            
            if (isempty(cutoff)), cutoff = q3; end
            
            cutoff = exp(cutoff) - 0.1;
        end
        
    end
    
    methods (Access = public)
        
        function this = kMRCD(kModel, NameValueArgs)
            arguments
                kModel
                NameValueArgs.Estimators
                NameValueArgs.cutoffEstimator (1,1) string {mustBeMember(NameValueArgs.cutoffEstimator, {'lognormal' 'chisquare' 'skewedbox'})} = 'lognormal';
            end
            
            if ~isempty(kModel)
                this.kModel = kModel;
            else
                this.kModel = LinKernel();
            end
            
            this.cutoffEstimator = NameValueArgs.cutoffEstimator;
            
            if isfield(NameValueArgs,"Estimators")
                this.estimators = NameValueArgs.Estimators;
            end
        end
        
        function solution = runAlgorithm(this, x, alpha)
            arguments
                this kMRCD
                x
                alpha (1,1) double {mustBeInRange(alpha,0.5,1)}
            end
            
            K = this.kModel.compute(x, x);
            [n, p] = size(x);
            
            %   Grab observation ranking from initial estimators
            solution = struct();
            
            if ismember('SDO', this.estimators)
                tic;
                outlyingnessIndices = Utils.SDO(K,alpha);
                t = toc;
                fprintf("SDO: %0.4f sec\n", t);
                
                res = struct("name", 'SDO', "outlyingnessIndices", outlyingnessIndices);
                if isempty(fieldnames(solution))
                    solution = res;
                else
                    solution = [solution, res];
                end
            end
            
            if ismember('SpatialRank', this.estimators)
                tic;
                outlyingnessIndices = Utils.SpatialRank(K,alpha);
                t = toc;
                fprintf("SpatialRank: %0.4f sec\n", t);
                
                res = struct("name", 'SpatialRank', "outlyingnessIndices", outlyingnessIndices);
                if isempty(fieldnames(solution))
                    solution = res;
                else
                    solution = [solution, res];
                end
            end
            
            if ismember('SpatialMedian', this.estimators)
                tic;
                outlyingnessIndices = Utils.SpatialMedianEstimator(K,alpha);
                t = toc;
                fprintf("SpatialMedian: %0.4f sec\n", t);
                
                res = struct("name", 'SpatialMedian', "outlyingnessIndices", outlyingnessIndices);
                if isempty(fieldnames(solution))
                    solution = res;
                else
                    solution = [solution, res];
                end
            end
            
            if ismember('SSCM', this.estimators)
                tic;
                outlyingnessIndices = Utils.SSCM(K);
                t = toc;
                fprintf("SSCM: %0.4f sec\n", t);
                
                res = struct("name", 'SSCM', "outlyingnessIndices", outlyingnessIndices);
                if isempty(fieldnames(solution))
                    solution = res;
                else
                    solution = [solution, res];
                end
            end
            
            scfac = Utils.MCDcons(p, alpha);
            
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
                    %disp('Running grid search');
                    grid = linspace(0.000001,1-0.000001,1000);
                    objgrid = abs(arrayfun(fncond,grid));
                    rhoL(index) = min(grid(objgrid == min(objgrid)));
                end
            end
            
            %   Set rho as max of the rho_i's obtained for each subset in previous step
            rho = max(rhoL(rhoL <= max([0.1, median(rhoL)])));
            
            %   Refine each initial estimation with C-steps
            Ktt_diag = diag(K);
            for index=1:numel(solution)
                disp(['Running C-Steps for ' solution(index).name '...']);
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
                    if isempty(setdiff(hSubset, solution(index).hsubsetIndices))
                        disp(['Convergence at iteration ' mat2str(iteration) ', ' solution(index).name]);
                        sigma = svd(Kc);
                        sigma = (1-rho)*scfac*sigma + numel(solution(index).hsubsetIndices)*rho;
                        solution(index).obj = sum(log(sigma));
                        solution(index).smd = smd;
                        break;
                    end
                end
                assert(iteration<this.cStepIterationsAllowed, 'no C-step convergence');
            end
            
            %   Select the solution with the lowest objective function
            [~, mIndex] = min([solution.obj]);
            %   ...and remove the other solutions
            solution = solution(mIndex);
            disp(['-> Best estimator is ' solution.name]);
            
            solution.rho = rho;
            solution.scfac = scfac;
            
            % Outlier flagging procedure
            solution.rd = max(sqrt(solution.smd),0);
            solution.ld = log(0.1 + solution.rd);
            
            if strcmp(this.cutoffEstimator, 'lognormal')
                solution.cutoff = this.lognormalCutoff(solution.rd, numel(solution.hsubsetIndices));
            elseif strcmp(this.cutoffEstimator, 'skewedbox')
                solution.cutoff = this.skewedboxCutoff(solution.rd);
            elseif strcmp(this.cutoffEstimator, 'chisquare')
                solution.cutoff = this.chisquareCutoff(solution.rd);
            end
            
            solution.flaggedOutlierIndices = find(solution.rd > solution.cutoff);
        end
        
    end
    
end

