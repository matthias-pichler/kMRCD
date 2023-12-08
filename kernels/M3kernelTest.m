classdef M3kernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity3(testCase)
            expected = [[1,   1/4, 1/4]
                        [1/4, 1  , 1/4]
                        [1/4, 1/4, 1  ]];
            x = eye(3);

            res = M3Kernel(x).compute(x);
            testCase.verifyEqual(res, expected, AbsTol=1/10000);
        end
    end
    
end