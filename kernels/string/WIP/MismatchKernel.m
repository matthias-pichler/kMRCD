classdef MismatchKernel < handle
    % see: https://string-kernel.readthedocs.io/en/latest/mismatch.html

    properties (Constant, Access = private)
        strkernelModule = py.importlib.import_module('strkernel.mismatch_kernel')
    end

    properties (SetAccess = private, GetAccess = public)
        subsequenceLength (1,1) uint32
        maxMismatches (1,1) uint32
        alphabetSize (1,1) uint32
    end

    methods (Access = private)
        function d = mismatchdist(this, ZI, ZJ)
            arguments
                this MismatchKernel
                ZI (1,1) string
                ZJ (:,1) string
            end

            kernel = this.strkernelModule.MismatchKernel(l=this.alphabetSize, k=this.subsequenceLength, m=this.maxMismatches);

            processed = this.strkernelModule.preprocess(py.list(cellstr([ZI' ZJ'])), ignoreLower=false);
            K = kernel.get_kernel(processed, normalize=true);
            K = double(K.kernel);

            d = K(2:end,1);
        end
    end
    
    methods (Access = public)
        
        function this = MismatchKernel(NameValueArgs)
            arguments
                NameValueArgs.subsequenceLength (1,1) double {mustBeInteger, mustBePositive}
                NameValueArgs.maxMismatches (1,1) double {mustBeInteger, mustBePositive}
                NameValueArgs.alphabetSize (1,1) double {mustBeInteger, mustBePositive}
            end
            
            this.alphabetSize = uint32(NameValueArgs.alphabetSize);
            this.subsequenceLength = uint32(NameValueArgs.subsequenceLength);
            this.maxMismatches = uint32(NameValueArgs.maxMismatches);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this MismatchKernel
                Xtrain (:, 1) string
                Xtest (:, 1) string = Xtrain
            end
            
            [nTrain, ~] = size(Xtrain);
            [nTest, ~] = size(Xtest);

            kernel = this.strkernelModule.MismatchKernel(l=this.alphabetSize, k=this.subsequenceLength, m=this.maxMismatches);

            processed = this.strkernelModule.preprocess(py.list(cellstr([Xtrain' Xtest'])), ignoreLower=false);
            res = kernel.get_kernel(processed, normalize=true);
            K = double(res.kernel);

            K = K(1:nTrain, (nTrain+1):end);
            
            assert(isequal(size(K), [nTrain, nTest]));
        end
    end
    
end

