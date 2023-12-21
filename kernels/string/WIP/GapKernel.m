classdef GapKernel < handle
    properties (SetAccess = private, GetAccess = public)
        lambda (1,1) double
        subsequenceLength (1,1) double
    end

    methods (Access = private)
        function g = gap(this, s, t)
            arguments
                this GapKernel
                s (1,1) string
                t (1,1) string
            end

            s = char(s);
            t = char(t);

            n = length(s);
            m = length(t);

            % Initialize the DPS matrix with zeros
            DPS = zeros(n, m);

            % Fill the DPS matrix according to the algorithm
            for i = 1:n
                for j = 1:m
                    if s(i) == t(j)
                        DPS(i, j) = this.lambda^2;
                    end
                end
            end

            p = this.subsequenceLength;

            K = zeros(1, p);

            % Initialize the DP matrix with zeros
            DP = zeros(n+1, m+1);

            % Calculate the DP values according to the algorithm
            for l = 2:p
                for i = 1:n-1
                    for j = 1:m-1
                        DP(i+1,j+1) = DPS(i,j) + this.lambda * DP(i,j+1) + ...
                                       this.lambda * DP(i+1,j) - this.lambda^2 * DP(i,j);
                        if s(i) == t(j)
                            DPS(i, j) = this.lambda^2 * DP(i, j);
                            K(l) = K(l) + DPS(i,j);
                        end
                    end
                end
            end

            disp(K)
        
            % Set the output to the last entry in K
            g = K(p);

        end
    end
    
    methods (Access = public)
        
        function this = GapKernel(NameValueArgs)
            arguments
                NameValueArgs.lambda (1,1) double {mustBeInRange(NameValueArgs.lambda, 0, 1)}
                NameValueArgs.subsequenceLength (1,1) double {mustBeInteger, mustBePositive}
            end
            
            this.lambda = NameValueArgs.lambda;
            this.subsequenceLength = NameValueArgs.subsequenceLength;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this GapKernel
                Xtrain (:, 1) string
                Xtest (:, 1) string = Xtrain
            end
            
            [nTrain, ~] = size(Xtrain);
            [nTest, ~] = size(Xtest);
            K = zeros(nTrain, nTest);

            for i=1:nTrain
                for j=1:nTest
                    K(i,j) = this.gap(Xtrain(i), Xtest(j));
                end
            end
            
            assert(isequal(size(K), [nTrain, nTest]));
        end
    end
    
end

