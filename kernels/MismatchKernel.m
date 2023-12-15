classdef MismatchKernel < handle
    % see: https://string-kernel.readthedocs.io/en/latest/mismatch.html

    properties (Constant, Access = private)
        strkernelModule = py.importlib.import_module('strkernel.mismatch_kernel')
    end

    properties (Access = private)
        kernel
    end
    
    methods (Access = public)
        
        function this = MismatchKernel(NameValueArgs)
            arguments
                NameValueArgs.subsequenceLength (1,1) double {mustBeInteger, mustBePositive}
                NameValueArgs.maxMismatches (1,1) double {mustBeInteger, mustBePositive}
                NameValueArgs.alphabetSize (1,1) double {mustBeInteger, mustBePositive}
            end
            
            alphabetSize = uint32(NameValueArgs.alphabetSize);
            subsequenceLength = uint32(NameValueArgs.subsequenceLength);
            maxMismatches = uint32(NameValueArgs.maxMismatches);

            this.kernel = this.strkernelModule.MismatchKernel(l=alphabetSize, k=subsequenceLength, m=maxMismatches);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this MismatchKernel
                Xtrain (1, :) string
                Xtest (1, :) string = Xtrain
            end

            processed = this.strkernelModule.preprocess(py.list(cellstr([Xtrain, Xtest])), ignoreLower=false);
            K = this.kernel.get_kernel(processed);
            K = double(K.kernel);

            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

