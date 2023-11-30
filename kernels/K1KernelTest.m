classdef K1KernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity2(testCase)
            expected = [[1/2,0]
                        [0,1/2]];

            res = K1Kernel(eye(2)).compute(eye(2));
            testCase.verifyEqual(res, expected, AbsTol=1/10000);
        end

        function identity3(testCase)
            expected = [[4/9,1/9,1/9]
                        [1/9,4/9,1/9]
                        [1/9,1/9,4/9]];

            res = K1Kernel(eye(3)).compute(eye(3));
            testCase.verifyEqual(res, expected, AbsTol=1/10000);
        end
    end
    
end