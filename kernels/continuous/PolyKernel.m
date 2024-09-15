classdef PolyKernel < KernelModel
    
    properties (Access = private)
        degree (1,1) double {mustBePositive, mustBeInteger} = 3;
    end
    
    methods (Access = public)
        
        function this = PolyKernel(degree)
            arguments
                degree (1,1) double {mustBePositive, mustBeInteger}
            end
            
            this.degree = degree;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this PolyKernel
                Xtrain (:,:) double
                Xtest (:,:) double = Xtrain
            end
            
            K = (Xtrain * Xtest' + 1).^this.degree;
        end
    end
    
end

