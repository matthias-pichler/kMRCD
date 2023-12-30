#include <vector>
#include <cmath>
#include <algorithm>

// Mex wrapper
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function
{
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs)
    {
        this->checkArguments(outputs, inputs);

        const matlab::data::TypedArray<double> Xtrain = inputs[0];
        const matlab::data::TypedArray<double> Xtest = inputs[1];
        const matlab::data::CellArray categories = inputs[2];
        const matlab::data::CellArray pmf = inputs[3];
        const double alpha = inputs[4][0];

        const auto K = this->m3_kernel(Xtrain, Xtest, categories, pmf, alpha);

        outputs[0] = std::move(K);
    }

private:
    matlab::data::ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = this->getEngine();

    void checkArguments(matlab::mex::ArgumentList &outputs, matlab::mex::ArgumentList &inputs)
    {
        // Check number of inputs
        if (inputs.size() != 5)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("5 inputs required")}));
        }

        auto Xtrain = inputs[0];
        auto Xtest = inputs[1];
        auto categories = inputs[2];
        auto pmf = inputs[3];
        auto alpha = inputs[4];

        // Check Xtrain argument: first input must be double (:,:)
        if (Xtrain.getType() != matlab::data::ArrayType::DOUBLE)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'Xtrain' must be double array of size (:,:)")}));
        }

        // Check Xtest argument: second input must be double (:,:)
        if (Xtest.getType() != matlab::data::ArrayType::DOUBLE)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'Xtest' must be double array of size (:,:)")}));
        }

        if (Xtrain.getDimensions().at(1) != Xtest.getDimensions().at(1))
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'Xtrain' and 'Xtest' must have the same number of columns")}));
        }

        if (Xtrain.getNumberOfElements() == 0 || Xtest.getNumberOfElements() == 0)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'Xtrain' and 'Xtest' must not be empty")}));
        }

        // Check categories argument: Third input must be cell (:,:)
        if (categories.getType() != matlab::data::ArrayType::CELL)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'categories' must be cell (:,:)")}));
        }

        if (categories.getDimensions().at(1) != Xtrain.getDimensions().at(1))
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'categories' must have the same number of columns as 'Xtrain' and 'Xtest'")}));
        }

        // Check pmf argument: Third input must be cell (:,:)
        if (pmf.getType() != matlab::data::ArrayType::CELL)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'pmf' must be cell (:,:)")}));
        }

        if (pmf.getDimensions().at(1) != Xtrain.getDimensions().at(1))
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'pmf' must have the same number of columns as 'Xtrain' and 'Xtest'")}));
        }

        // Check alpha argument: Forth input must be double (1,1)
        if (alpha.getType() != matlab::data::ArrayType::DOUBLE ||
            alpha.getNumberOfElements() != 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("'alpha' must be double (1,1)")}));
        }

        // Check number of outputs
        if (outputs.size() > 1)
        {
            this->matlabPtr->feval(u"error",
                                   0,
                                   std::vector<matlab::data::Array>({this->factory.createScalar("Only one output is returned")}));
        }
    }

    double invert(double z, double alpha)
    {
        return std::pow((1.0 - std::pow(z, alpha)), 1.0 / alpha);
    }

    double m3_dist(const matlab::data::TypedArray<double> &xs, const matlab::data::TypedArray<double> &ys, size_t i, size_t j, const matlab::data::CellArray &cs, const matlab::data::CellArray &p, double alpha)
    {

        size_t len = xs.getDimensions().at(1);
        double numerator = 0.0, denominator = 0.0;

        auto categories = matlab::data::CellArray(cs);
        auto pmf = matlab::data::CellArray(p);

        for (size_t column = 0; column < len; ++column)
        {
            const matlab::data::TypedArrayRef<double> currentCategories = categories[column];
            const matlab::data::TypedArrayRef<double> currentPmf = pmf[column];
            auto xCategoryIterator = std::find(currentCategories.begin(), currentCategories.end(), xs[i][column]);
            auto yCategoryIterator = std::find(currentCategories.begin(), currentCategories.end(), ys[j][column]);

            if (xCategoryIterator == currentCategories.end() || yCategoryIterator == currentCategories.end())
            {
                throw std::runtime_error("Input arrays must have the same categories.");
            }

            size_t xCategoryIndex = std::distance(currentCategories.begin(), xCategoryIterator);
            size_t yCategoryIndex = std::distance(currentCategories.begin(), yCategoryIterator);

            const double pX = currentPmf[xCategoryIndex];
            const double pY = currentPmf[yCategoryIndex];

            if (xs[i][column] == ys[j][column])
            {
                numerator += this->invert(pX, alpha);
            }
            denominator += this->invert(pX, alpha) + this->invert(pY, alpha);
        }

        return 2.0 * numerator / denominator;
    }

    matlab::data::TypedArray<double> m3_kernel(const matlab::data::TypedArray<double> &xs, const matlab::data::TypedArray<double> &ys, const matlab::data::CellArray &categories, const matlab::data::CellArray &pmf, double alpha)
    {

        size_t heightX = xs.getDimensions().at(0);
        size_t heightY = ys.getDimensions().at(0);

        matlab::data::TypedArray<double> K = this->factory.createArray<double>({heightX, heightY});

        for (size_t i = 0; i < heightX; ++i)
        {
            for (size_t j = 0; j < heightY; ++j)
            {
                K[i][j] = m3_dist(xs, ys, i, j, categories, pmf, alpha);
            }
        }

        return K;
    }
};
