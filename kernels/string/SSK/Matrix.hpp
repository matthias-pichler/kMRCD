#ifndef MATRIX_H
#define MATRIX_H

#include <array>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

template <typename T, size_t N>
class Matrix
{
public:
    Matrix(const std::array<size_t, N> &dimensions) : dims(dimensions)
    {
        initialize();
    }

    Matrix(std::initializer_list<size_t> dimensions_list)
    {
        if (dimensions_list.size() != N)
        {
            throw std::invalid_argument("Number of dimensions must be " + std::to_string(N));
        }
        std::copy(dimensions_list.begin(), dimensions_list.end(), dims.begin());
        initialize();
    }

    inline T &at(const std::array<size_t, N> &indices)
    {
        size_t index = compute_flat_index(indices);
        return data[index];
    }

    inline const T &at(const std::array<size_t, N> &indices) const
    {
        size_t index = compute_flat_index(indices);
        return data[index];
    }

    inline T &at(std::initializer_list<size_t> indices_list)
    {
        if (indices_list.size() != N)
        {
            throw std::invalid_argument("Number of indices must be " + std::to_string(N));
        }
        std::array<size_t, N> indices;
        std::copy(indices_list.begin(), indices_list.end(), indices.begin());
        return at(indices);
    }

    inline const T &at(std::initializer_list<size_t> indices_list) const
    {
        if (indices_list.size() != N)
        {
            throw std::invalid_argument("Number of indices must be " + std::to_string(N));
        }
        std::array<size_t, N> indices;
        std::copy(indices_list.begin(), indices_list.end(), indices.begin());
        return at(indices);
    }

    // to_string method for 2D matrices
    std::string to_string() const
    {
        if (N != 2)
        {
            throw std::logic_error("to_string() is only supported for 2D matrices");
        }

        std::ostringstream oss;
        size_t rows = dims[0];
        size_t cols = dims[1];

        for (size_t i = 0; i < rows; i++)
        {
            oss << "[ ";
            for (size_t j = 0; j < cols; j++)
            {
                oss << std::setw(8) << at({i, j}) << " ";
            }
            oss << "]\n";
        }
        return oss.str();
    }

    inline size_t size() const
    {
        return total_size;
    }

private:
    std::array<size_t, N> dims;    // Dimensions of the matrix
    std::array<size_t, N> strides; // Strides for each dimension
    size_t total_size;             // Total number of elements
    std::vector<T> data;           // Flat data storage

    // Common initialization function
    void initialize()
    {
        total_size = 1;
        for (int i = static_cast<int>(N) - 1; i >= 0; i--)
        {
            strides[i] = total_size;
            total_size *= dims[i];
        }
        data.resize(total_size);
    }

    // Compute the flat index from multi-dimensional indices
    inline size_t compute_flat_index(const std::array<size_t, N> &indices) const
    {
        size_t index = 0;
        for (size_t i = 0; i < N; i++)
        {
            if (indices[i] >= dims[i])
            {
                throw std::out_of_range("Index out of bounds in dimension " + std::to_string(i));
            }
            index += indices[i] * strides[i];
        }
        return index;
    }
};

#endif // MATRIX_H
