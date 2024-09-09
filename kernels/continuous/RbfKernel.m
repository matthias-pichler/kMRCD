classdef RbfKernel < handle
    
    properties (Access = public)
        sigma (1,1) double {mustBePositive} = 1;
    end
    
    methods (Access = public)
        
        function this = RbfKernel(bandwidth)
            arguments
                bandwidth (1,1) double {mustBePositive}
            end
            
            this.sigma = bandwidth;
        end
        
        function updateKernel(this, bandwidth)
            arguments
                this RbfKernel
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
            
            n=size(Xtrain, 1);
            m=size(Xtest, 1);
            Ka = repmat(sum(Xtrain.^2,2), 1, m);
            Kb = repmat(sum(Xtest.^2,2), 1, n);
            K = (Ka + Kb' - 2 .* (Xtrain * Xtest'));
            K = exp(-K ./ (2* this.sigma^2));
        end
    end
    
end

