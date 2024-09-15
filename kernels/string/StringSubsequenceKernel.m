classdef StringSubsequenceKernel < KernelModel

    properties (GetAccess = public, SetAccess = private)
        lambda (1,1) double {mustBeInRange(lambda, 0, 1)} = 0.5
        maxSubsequence (1,1) double {mustBeInteger, mustBePositive} = 1
    end

    methods (Access = public)
        
        function this = StringSubsequenceKernel(NameValueArgs)
            arguments
                NameValueArgs.lambda (1,1) double
                NameValueArgs.maxSubsequence (1,1) double
            end
            
            this.lambda = NameValueArgs.lambda;
            this.maxSubsequence = NameValueArgs.maxSubsequence;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this StringSubsequenceKernel
                Xtrain (:, 1) string
                Xtest (:, 1) string = Xtrain
            end
            
            K = ssk(Xtrain, Xtest, uint8(this.maxSubsequence), this.lambda);

            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

