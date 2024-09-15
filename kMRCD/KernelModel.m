classdef (Abstract) KernelModel < handle
  methods(Abstract, Access = public)
    K = compute(this, Xtrain, Xtest)
  end
end