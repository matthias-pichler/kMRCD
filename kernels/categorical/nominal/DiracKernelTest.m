classdef DiracKernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity2(testCase)
            res = DiracKernel().compute(eye(2));

            testCase.verifyEqual(res, eye(2));
        end

        function identity3(testCase)
            expected = [[1,1/3,1/3];[1/3,1,1/3];[1/3,1/3,1]];

            res = DiracKernel().compute(eye(3));

            testCase.verifyEqual(res, expected);
        end

        function identity4(testCase)
            expected = [[1,   1/2, 1/2, 1/2]
                        [1/2, 1,   1/2, 1/2]
                        [1/2, 1/2, 1,   1/2]
                        [1/2, 1/2, 1/2, 1]];

            res = DiracKernel().compute(eye(4));

            testCase.verifyEqual(res, expected);
        end

        function diagonal(testCase)
            res = DiracKernel().compute(rand(3));
            
            testCase.verifyEqual(diag(res), [1,1,1]');
        end
    end
    
end