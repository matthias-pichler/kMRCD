classdef GapKernelTest < matlab.unittest.TestCase

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
            x = "gatta";
            y = "cata";
            expected = 6 * l^2;

            kernel = GapKernel(lambda=l, subsequenceLength=1);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

        function test2(testCase)
            l = 0.6;
            x = "gatta";
            y = "cata";
            expected = l^7 + 2*l^5 + 2*l^4;

            kernel = GapKernel(lambda=l, subsequenceLength=2);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

        function test3(testCase)
            l = 1;
            x = "gatta";
            y = "cata";
            expected = 2*l^7;

            kernel = GapKernel(lambda=l, subsequenceLength=3);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=1/100000);
        end

    end

end