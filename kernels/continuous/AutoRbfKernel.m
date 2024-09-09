classdef AutoRbfKernel < handle
    
    properties (GetAccess = public, SetAccess = private)
        sigma (1,1) double {mustBePositive} = 1
    end

    methods (Access = private)

        function sigma = medianBandwidth(this, x)
            arguments
                this AutoRbfKernel
                x double
            end

            distances = pdist(x, "squaredeuclidean");
            sigma = sqrt(median(distances));
        end

        function sigma = meanBandwidth(this, x)
            arguments
                this AutoRbfKernel
                x double
            end

            % see Chaudhuri et al. 2017
            N = height(x);
            delta = sqrt(2) * 10^-6;

            s2 = var(x);

            sigma = sqrt( 2 * N * sum(s2)/((N-1) * log((N-1)/delta^2)) );
        end

        function sigma = modifiedmeanBandwidth(this, x)
            arguments
                this AutoRbfKernel
                x double
            end

            % see Liao et al. 2018
            N = height(x);
            phi = 1 / log(N - 1);
            delta = -0.14818008* phi^4 + 0.284623624 * phi^3 - 0.252853808 * phi^2 + 0.159059498 * phi - 0.001381145;

            s2 = var(x);

            sigma = sqrt( 2 * N * sum(s2)/((N-1) * log((N-1)/delta^2)) );
        end

    end
    
    methods (Access = public)
        
        function this = AutoRbfKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.bandwidth (1,1) string {mustBeMember(NameValueArgs.bandwidth, {'median' 'mean' 'modifiedmean'})} = 'median';
            end

            if strcmp(NameValueArgs.bandwidth, 'median')
                this.sigma = this.medianBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'mean')
                this.sigma = this.meanBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'modifiedmean')
                this.sigma = this.modifiedmeanBandwidth(x);
            end
            
            distances = pdist(x, "squaredeuclidean");
            this.sigma = sqrt(median(distances));
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this AutoRbfKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            % K(x,y) = k(||x-y||) = exp(-1/2s^2 * ||x-y||^2)
            % ||x-y||^2 = sum((x_i - y_i)^2) = sum(x_i^2-2*x_i*y_i+y_i)
            %           = sum(x_i^2) + sum(y_i^2) - 2*sum(x_i*y_i)

            n=size(Xtrain, 1);
            m=size(Xtest, 1);
            Ka = repmat(sum(Xtrain.^2,2), 1, m); % sum(x_i^2)
            Kb = repmat(sum(Xtest.^2,2), 1, n); % sum(y_i^2)
            K = (Ka + Kb' - 2 .* (Xtrain * Xtest')); % x_i^2+y_i-2*x_i*y_i = x_i^2-2*x_i*y_i+y_i
            K = exp(-K ./ (2* this.sigma^2));
        end
    end
    
end

