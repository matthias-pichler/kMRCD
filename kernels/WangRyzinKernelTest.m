classdef WangRyzinKernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity2(testCase)
            l = 0.75;
            x = eye(2) * 2;

            expected = repmat((0.5*l*(1-l)^2)^2, size(x));
            expected(logical(eye(size(expected)))) = l^2;

            kernel = WangRyzinKernel(x);
            kernel.lambda = [l l];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity3(testCase)
            l = 0.75;
            x = eye(3) * 4;

            expected = repmat(l*(0.5*l*(1-l)^4)^2, size(x));
            expected(logical(eye(size(expected)))) = l^3;

            kernel = WangRyzinKernel(x);
            kernel.lambda = [l l l];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity4(testCase)
            l = 0.75;
            x = eye(4);
            
            expected = repmat(l^2*(0.5*l*(1-l))^2, size(x));
            expected(logical(eye(size(expected)))) = l^4;

            kernel = WangRyzinKernel(x);
            kernel.lambda = [l l l l];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end