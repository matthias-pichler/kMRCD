classdef AutoSphereRbfKernel < handle
    
    properties (Access = public)
        sigma;
    end
    
    methods (Access = public)
        
        function this = AutoSphereRbfKernel(x)
            arguments
                x double
            end
            
            % TODO: is this still the best way to estimate sigma?
            distances = pdist(x, "squaredeuclidean");
            this.sigma = sqrt(median(distances));
            disp(['AutoSphereRbfKernel: Sigma = ' mat2str(this.sigma)]);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this
                Xtrain double
                Xtest double = Xtrain
            end
            
            % K(x,y) = k(x*y) = exp(-2/s*(1-x*y))
            
            K = 1 - (Xtrain * Xtest');
            K = exp(-(2/this.sigma) * K);
        end
    end
    
end

