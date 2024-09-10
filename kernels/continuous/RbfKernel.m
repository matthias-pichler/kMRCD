classdef RbfKernel < handle
    
    properties (GetAccess = public, SetAccess = protected)
        sigma (1,1) double {mustBePositive} = 1;
    end
    
    methods (Access = public)
        
        function this = RbfKernel(bandwidth)
            arguments
                bandwidth (1,1) double {mustBePositive}
            end
            
            this.sigma = bandwidth;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this RbfKernel
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

