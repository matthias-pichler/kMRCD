classdef OrderedLiRacinKernelTest < matlab.unittest.TestCase
    
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

            expected = repmat(l^(2*2), size(x));
            expected(logical(eye(size(expected)))) = 1;

            kernel = OrderedLiRacinKernel(x, lambda=[l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity2lambda(testCase)
            l1 = 0.75;
            l2 = 0.5;

            x = eye(2) * 2;

            expected = [[1,         l1^2*l2^2]
                        [l1^2*l2^2, 1]];

            kernel = OrderedLiRacinKernel(x, lambda=[l1 l2]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity3(testCase)
            l = 0.75;
            x = eye(3) * 4;

            expected = repmat(l^(2*4), size(x));
            expected(logical(eye(size(expected)))) = 1;

            kernel = OrderedLiRacinKernel(x, lambda=[l l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity4(testCase)
            l = 0.75;
            x = eye(4);
            
            expected = repmat(l^2, size(x));
            expected(logical(eye(size(expected)))) = 1;

            kernel = OrderedLiRacinKernel(x, lambda=[l l l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end