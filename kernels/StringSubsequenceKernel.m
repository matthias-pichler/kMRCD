classdef StringSubsequenceKernel < handle

    properties (GetAccess = public, SetAccess = private)
        lambda (1,1) double {mustBeInRange(lambda, 0, 1)} = 0.5
        maxSubsequence (1,1) double {mustBeInteger, mustBePositive} = 1
    end
    
    methods (Access = private)
        function k = ssk(this, s, t)
            arguments
                this StringSubsequenceKernel
                s (1,1) string
                t (1,1) string
            end

            % to byte codes
            s = double(char(s));
            t = double(char(t));

            k_prim = zeros(this.maxSubsequence, length(s), length(t));
            k_prim(1,:,:) = 1;
        
            for i = 2:this.maxSubsequence
                for sj = i:length(s)
                    toret = 0;
                    for tk = i:length(t)
                        if s(sj-1) == t(tk-1)
                            toret = this.lambda * (toret + this.lambda * k_prim(i-1, sj-1, tk-1));
                        else
                            toret = toret * this.lambda;
                        end
                        k_prim(i, sj, tk) = toret + this.lambda * k_prim(i, sj-1, tk);
                    end
                end
            end
 
            k = 0;
            for i = 1:this.maxSubsequence
                for sj = i:length(s)
                    for tk = i:length(t)
                        if s(sj) == t(tk)
                            k = k + this.lambda^2 * k_prim(i, sj, tk);
                        end
                    end
                end
            end
        end

        function d = sskdist(this, ZI, ZJ)
            arguments
                this StringSubsequenceKernel
                ZI (1, 1) string
                ZJ (:, 1) string
            end
        end
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
            
            [nTrain, ~] = size(Xtrain);
            [nTest, ~] = size(Xtest);

            K = zeros(nTrain, nTest);
        
            for i = 1:nTrain
                for j = 1:nTest
                    K(i, j) = this.ssk(Xtrain(i, :), Xtest(j, :));
                end
            end
        
            K_train = zeros(nTrain, 1);
            K_test = zeros(nTest, 1);
        
            for i = 1:nTrain
                K_train(i) = this.ssk(Xtrain(i, :), Xtrain(i, :));
            end
            for j = 1:nTest
                K_test(j) = this.ssk(Xtest(j, :), Xtest(j, :));
            end
        
            K = K ./ sqrt(repmat(K_test', height(K_train), 1) .* K_train);

            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

