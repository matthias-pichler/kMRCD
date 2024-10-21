#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include "Matrix.hpp"

// Mex wrapper
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function
{
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs)
    {
        this->checkArguments(outputs, inputs);

        const matlab::data::StringArray Xtrain = inputs[0];
        const matlab::data::StringArray Xtest = inputs[1];
        const size_t n = inputs[2][0];
        const double lambda = inputs[3][0];

        const auto K = this->string_kernel(Xtrain, Xtest, n, lambda);

        outputs[0] = std::move(K);
    }

private:
    matlab::data::ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = this->getEngine();

    void checkArguments(matlab::mex::ArgumentList &outputs, matlab::mex::ArgumentList &inputs)
    {
        // Check number of inputs
        if (inputs.size() != 4)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("Four inputs required")}));
        }

        // Check Xtrain argument: first input must be string (:,1)
        if (inputs[0].getType() != matlab::data::ArrayType::MATLAB_STRING ||
            inputs[0].getDimensions().at(1) != 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'Xtrain' must be string array of size (:,1)")}));
        }

        // Check Xtest argument: second input must be string (:,1)
        if (inputs[1].getType() != matlab::data::ArrayType::MATLAB_STRING ||
            inputs[1].getDimensions().at(1) != 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'Xtest' must be string array of size (:,1)")}));
        }

        // Check maxSubsequence argument: Third input must be scalar uint8
        if (inputs[2].getType() != matlab::data::ArrayType::UINT8 ||
            inputs[2].getNumberOfElements() != 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'maxSubsequence' must be scalar uint8")}));
        }

        // Check lambda argument: Third input must be scalar double
        if (inputs[3].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[3].getNumberOfElements() != 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'lambda' must be scalar double")}));
        }

        // Check number of outputs
        if (outputs.size() > 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("Only one output is returned")}));
        }
    }

    matlab::data::TypedArray<double> string_kernel(const matlab::data::StringArray &xs, const matlab::data::StringArray &ys, size_t n, double lambda)
    {
        size_t lenxs = xs.getDimensions().at(0), lenys = ys.getDimensions().at(0);

        if (lenxs == 0 || lenys == 0)
        {
            throw std::runtime_error("Input strings must not be empty.");
        }

        matlab::data::TypedArray<double> mat = this->factory.createArray<double>({lenxs, lenys});

        for (size_t i = 0; i < lenxs; ++i)
        {
            for (size_t j = 0; j < lenys; ++j)
            {
                mat[i][j] = ssk(xs[i], ys[j], n, lambda);
            }
        }

        matlab::data::TypedArray<double> mat_xs = this->factory.createArray<double>({lenxs}, {0.0});
        matlab::data::TypedArray<double> mat_ys = this->factory.createArray<double>({lenys}, {0.0});

        for (size_t i = 0; i < lenxs; ++i)
        {
            mat_xs[i] = ssk(xs[i], xs[i], n, lambda);
        }
        for (size_t j = 0; j < lenys; ++j)
        {
            mat_ys[j] = ssk(ys[j], ys[j], n, lambda);
        }

        for (size_t i = 0; i < lenxs; ++i)
        {
            for (size_t j = 0; j < lenys; ++j)
            {
                mat[i][j] /= sqrt(mat_xs[i] * mat_ys[j]);
            }
        }

        return mat;
    }

    double ssk(const matlab::data::MATLABString &s_, const matlab::data::MATLABString &t_, size_t p, double lambda)
    {
        std::u16string s = *s_;
        std::u16string t = *t_;

        size_t n = s.length();
        size_t m = t.length();

        auto k_prim = new double[p * n * m]{};
        auto k_prim_idx = [&](size_t i, size_t j, size_t k)
        { return i * n * m + j * m + k; };

        std::fill_n(k_prim, (n * m), 1); // k_prim[0][*][*] = 1

        for (size_t i = 1; i < p; ++i)
        {
            for (size_t s_j = i; s_j < n; ++s_j)
            {
                double toret = 0.0;
                for (size_t t_k = i; t_k < m; ++t_k)
                {
                    if (s[s_j - 1] == t[t_k - 1])
                    {
                        toret = lambda * (toret + lambda * k_prim[k_prim_idx(i - 1, s_j - 1, t_k - 1)]);
                    }
                    else
                    {
                        toret *= lambda;
                    }
                    k_prim[k_prim_idx(i, s_j, t_k)] = toret + lambda * k_prim[k_prim_idx(i, s_j - 1, t_k)];
                }
            }
        }

        double k = 0.0;
        for (size_t i = 0; i < p; ++i)
        {
            for (size_t s_j = i; s_j < n; ++s_j)
            {
                for (size_t t_k = i; t_k < m; ++t_k)
                {
                    if (s[s_j] == t[t_k])
                    {
                        k += lambda * lambda * k_prim[k_prim_idx(i, s_j, t_k)];
                    }
                }
            }
        }

        delete[] k_prim;

        return k;
    }
};
