classdef DiracKernel < handle
    %
    %               n
    % k(x,y) = 1/n Sum x_i == y_i
    %              i=1
    %

    methods (Static, Access = private)
        function d = diracdist(ZI, ZJ)
            arguments
                ZI (1,:) double
                ZJ double
            end

            d = mean(ZI == ZJ, 2);
        end
    end

    methods (Access = public)
        function K = compute(~, Xtrain, Xtest)
            arguments
                ~
                Xtrain (:, :) double
                Xtest (:, :) double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @DiracKernel.diracdist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

