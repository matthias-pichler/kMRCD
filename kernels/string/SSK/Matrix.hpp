#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <string>
#include <sstream>
#include <iomanip> // For formatting output

template <typename T, size_t N>
class Matrix
{
public:
    Matrix(const std::vector<size_t> &dimensions)
    {
        std::copy(dimensions.begin(), dimensions.end(), dims.begin());
        initialize();
    }

    Matrix(size_t&&dimensions_list) : dims{{std::forward<size_t>(dimensions_list)}}
    {
        initialize();
    }

    T &at(const std::array<size_t, N> &indices)
    {
        size_t index = compute_flat_index(indices);
        return data[index];
    }

    const T &at(const std::array<size_t, N> &indices) const
    {
        size_t index = compute_flat_index(indices);
        return data[index];
    }

    T &at(size_t&&indices_list)
    {
        std::array<size_t, N> indices{{std::forward<size_t>(indices_list)}};
        return at(indices);
    }

    const T &at(size_t&&indices_list) const
    {
        std::array<size_t, N> indices{{std::forward<size_t>(indices_list)}};
        return at(indices);
    }

    const std::array<size_t, N> &dimensions() const
    {
        return dims;
    }

    size_t size() const
    {
        return total_size;
    }

    std::string to_string() const {
        if (dims.size() != 2) {
            throw std::logic_error("to_string() is only supported for 2D matrices");
        }

        std::ostringstream oss;
        size_t rows = dims[0];
        size_t cols = dims[1];

        for (size_t i = 0; i < rows; ++i) {
            oss << "[ ";
            for (size_t j = 0; j < cols; ++j) {
                oss << std::setw(8) << get({i, j}) << " ";
            }
            oss << "]\n";
        }
        return oss.str();
    }

private:
    std::array<size_t, N> dims;  // Dimensions of the matrix
    std::array<size_t, N> strides; // Strides for each dimension
    size_t total_size;           // Total number of elements
    std::vector<T> data;         // Flat data storage

    void initialize()
    {
        if (dims.empty())
        {
            throw std::invalid_argument("Dimensions cannot be empty");
        }

        // Calculate total size and strides
        total_size = 1;
        for (int i = 0; i < N; i++)
        {
            strides[i] = total_size;
            total_size *= dims[i];
        }

        // Initialize the flat data vector
        data.resize(total_size);
    }

    // Compute the flat index from multi-dimensional indices
    inline size_t compute_flat_index(const std::array<size_t, N> &indices) const
    {
        if (indices.size() != N)
        {
            throw std::invalid_argument("Number of indices must match number of dimensions");
        }

        size_t index = 0;
        for (size_t i = 0; i < N; i++)
        {
            if (indices[i] >= dims[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * dims[i];
        }
        return index;
    }
};

#endif // MATRIX_HPP
