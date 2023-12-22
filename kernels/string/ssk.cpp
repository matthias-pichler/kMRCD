#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <iostream>

// Kernel defined by Lodhi et al. (2002)
double ssk(const std::string& s_, const std::string& t_, size_t n, double lbda) {
    size_t lens = s_.length(), lent = t_.length();
    std::vector<std::vector<std::vector<double>>> k_prim(n, std::vector<std::vector<double>>(lens, std::vector<double>(lent, 0)));

    for (size_t i = 0; i < lens; ++i) {
        for (size_t j = 0; j < lent; ++j) {
            k_prim[0][i][j] = 1;
        }
    }

    for (size_t i = 1; i < n; ++i) {
        for (size_t sj = i; sj < lens; ++sj) {
            double toret = 0.0;
            for (size_t tk = i; tk < lent; ++tk) {
                if (s_[sj-1] == t_[tk-1]) {
                    toret = lbda * (toret + lbda * k_prim[i-1][sj-1][tk-1]);
                } else {
                    toret *= lbda;
                }
                k_prim[i][sj][tk] = toret + lbda * k_prim[i][sj-1][tk];
            }
        }
    }

    double k = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t sj = i; sj < lens; ++sj) {
            for (size_t tk = i; tk < lent; ++tk) {
                if (s_[sj] == t_[tk]) {
                    k += lbda * lbda * k_prim[i][sj][tk];
                }
            }
        }
    }
    return k;
}

std::vector<std::vector<double>> string_kernel(const std::vector<std::string>& xs, const std::vector<std::string>& ys, size_t n, double lbda) {
    size_t lenxs = xs.size(), lenys = ys.size();

    if (lenxs == 0 || lenys == 0) {
        throw std::runtime_error("Input strings must not be empty.");
    }

    std::vector<std::vector<double>> mat(lenxs, std::vector<double>(lenys, 0.0));

    for (size_t i = 0; i < lenxs; ++i) {
        for (size_t j = 0; j < lenys; ++j) {
            mat[i][j] = ssk(xs[i], ys[j], n, lbda);
        }
    }

    std::vector<double> mat_xs(lenxs, 0.0);
    std::vector<double> mat_ys(lenys, 0.0);

    for (size_t i = 0; i < lenxs; ++i) {
        mat_xs[i] = ssk(xs[i], xs[i], n, lbda);
    }
    for (size_t j = 0; j < lenys; ++j) {
        mat_ys[j] = ssk(ys[j], ys[j], n, lbda);
    }

    for (size_t i = 0; i < lenxs; ++i) {
        for (size_t j = 0; j < lenys; ++j) {
            mat[i][j] /= sqrt(mat_xs[i] * mat_ys[j]);
        }
    }

    return mat;
}

// Example usage
int main() {
    std::vector<std::string> xs = {"cat", "car", "cart", "camp", "shard"};
    std::vector<std::string> ys = {"a", "cd"};
    size_t n = 2;
    double lbda = 1.0;

    std::vector<std::string> test = {"This is a very long string, just to test how fast this implementation of ssk is. It should look like the computation tooks no time, unless you're running this in a potato pc"};

    try {
        auto result = string_kernel(xs, ys, n, lbda);
        // Output or process the result as needed
        for (const auto& row : result) {
            for (const auto& element : row) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(10) << element << " ";
            }
            std::cout << std::endl;
        }

        string_kernel(test, test, 30, 0.8f);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
