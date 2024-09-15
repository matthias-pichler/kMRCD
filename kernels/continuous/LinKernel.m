classdef LinKernel < KernelModel
    
    methods (Access = public)
        function K = compute(~, Xtrain, Xtest)
            arguments
                ~
                Xtrain (:, :) double
                Xtest (:, :) double = Xtrain
            end
            
            K = Xtrain * Xtest';
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

