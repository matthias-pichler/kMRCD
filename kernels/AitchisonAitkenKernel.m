classdef AitchisonAitkenKernel < handle
    
    properties (Access = public)
        lambda (1,1) double {mustBeInRange(lambda,0.5,1)} = 1
    end
    
    methods (Access = public)
        
        function this = AitchisonAitkenKernel(x)
            arguments
                x double
            end
            
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this AitchisonAitkenKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
        end
    end
    
end

