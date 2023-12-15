classdef MismatchKernelTest < matlab.unittest.TestCase

    methods (TestClassSetup)
        % Shared setup for the entire test class
        function setup(testCase)
            projectDir = fileparts(fileparts(which(mfilename)));
            pyenv(Version=fullfile(projectDir, ".venv", "bin", "python"), ExecutionMode='OutOfProcess');
        end
    end

    methods (TestMethodSetup)
        % Setup for each test
    end

    methods (Test)
        % Test methods

        function diagonalis1(testCase)
            x = ["foo", "bar"];

            kernel = MismatchKernel(alphabetSize=26, subsequenceLength=3, maxMismatches=1);

            res = kernel.compute(x);

            testCase.verifyEqual(diag(res), ones(2,1));
        end
    end

end