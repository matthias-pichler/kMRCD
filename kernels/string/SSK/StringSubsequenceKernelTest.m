classdef StringSubsequenceKernelTest < matlab.unittest.TestCase

    methods (TestClassSetup)
        % Shared setup for the entire test class
    end

    methods (TestMethodSetup)
        % Setup for each test
    end

    methods (Test)
        % Test methods

        function test1(testCase)
            l = 0.6;
            x = "cat";
            y = "cart";
            expected = 0.77774157;

            kernel = StringSubsequenceKernel(maxSubsequence=4, lambda=l);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

        function test2(testCase)
            l = 1;
            x = "science is organized knowledge";
            y = "wisdom is organized life";
            expected = 0.27714312;

            kernel = StringSubsequenceKernel(maxSubsequence=4, lambda=l);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

        function test3(testCase)
            l = 1;
            x = ["cat"; "car"; "cart"; "camp"; "shard"];
            y = ["a"; "cd"];
            expected = [[0.40824829 0.23570226]
                        [0.40824829 0.23570226]
                        [0.31622777 0.18257419]
                        [0.31622777 0.18257419]
                        [0.25819889 0.1490712 ]];

            kernel = StringSubsequenceKernel(maxSubsequence=2, lambda=l);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

        function test4(testCase)
            l = 0.8;
            x = "This is a very long string, just to test how fast this implementation of ssk is. It should look like the computation tooks no time, unless you're running this in a potato pc";
            expected = 1;

            kernel = StringSubsequenceKernel(maxSubsequence=30, lambda=l);

            res = kernel.compute(x, x);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end
    end

end