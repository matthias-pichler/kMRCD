classdef MismatchKernelTest < matlab.unittest.TestCase

    methods (TestClassSetup)
        % Shared setup for the entire test class
        function setup(testCase)
            fileDir = fileparts(which(mfilename));
            projectDir = fileparts(fileparts(fileDir));
            pyenv(Version=fullfile(projectDir, ".venv", "bin", "python"), ExecutionMode='OutOfProcess');
        end
    end

    methods (TestMethodSetup)
        % Setup for each test
    end

    methods (Test)
        % Test methods

        function diag1(testCase)
            x = ["foo"; "bar"];

            kernel = MismatchKernel(alphabetSize=26, subsequenceLength=3, maxMismatches=1);

            res = kernel.compute(x);

            testCase.verifyEqual(diag(res), ones(2,1));
        end

        function test1(testCase)
            l = 0.6;
            x = "cat";
            y = "cart";
            expected = 0.77774157;

            kernel = MismatchKernel(alphabetSize=26, subsequenceLength=4, maxMismatches=1);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

        function test2(testCase)
            l = 1;
            x = ["cat"; "car"; "cart"; "camp"; "shard"];
            y = ["a"; "cd"];
            expected = [[0.40824829 0.23570226]
                        [0.40824829 0.23570226]
                        [0.31622777 0.18257419]
                        [0.31622777 0.18257419]
                        [0.25819889 0.1490712 ]];

            kernel = MismatchKernel(alphabetSize=26, subsequenceLength=4, maxMismatches=1);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end
    end

end