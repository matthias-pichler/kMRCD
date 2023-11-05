classdef AutoLaplacianKernel < handle
    
    properties (Access = public)
        sigma;
    end
    
    methods (Access = public)
        
        function this = AutoLaplacianKernel(x)
            arguments
                x (:,:) double
            end
            
            distances = pdist(x).^2;
            this.sigma = sqrt(median(distances));
            disp(['AutoLaplacianKernel: Sigma = ' mat2str(this.sigma)]);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this
                Xtrain (:,:) double
                Xtest (:,:) double = Xtrain
            end
            
            n=size(Xtrain, 1);
            m=size(Xtest, 1);
            Ka = repmat(sum(abs(Xtrain),2), 1, m);
            Kb = repmat(sum(abs(Xtest),2), 1, n);
            K = (Ka + Kb');
            K = exp(-K ./ (this.sigma));
        end
    end
    
end
