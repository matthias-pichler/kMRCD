#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <initializer_list>

template <typename T>
class Matrix
{
public:
    Matrix(const std::vector<size_t> &dimensions) : dims(dimensions)
    {
        initialize();
    }

    Matrix(std::initializer_list<size_t> dimensions_list) : dims(dimensions_list)
    {
        initialize();
    }

    T &at(const std::vector<size_t> &indices)
    {
        size_t index = compute_flat_index(indices);
        return data[index];
    }

    const T &at(const std::vector<size_t> &indices) const
    {
        size_t index = compute_flat_index(indices);
        return data[index];
    }

    T &at(std::initializer_list<size_t> indices_list)
    {
        std::vector<size_t> indices(indices_list);
        return at(indices);
    }

    const T &at(std::initializer_list<size_t> indices_list) const
    {
        std::vector<size_t> indices(indices_list);
        return at(indices);
    }

    const std::vector<size_t> &dimensions() const
    {
        return dims;
    }

    size_t size() const
    {
        return total_size;
    }

private:
    std::vector<size_t> dims;    // Dimensions of the matrix
    std::vector<size_t> strides; // Strides for each dimension
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
        strides.resize(dims.size());
        for (int i = dims.size() - 1; i >= 0; --i)
        {
            strides[i] = total_size;
            total_size *= dims[i];
        }

        // Initialize the flat data vector
        data.resize(total_size);
    }

    // Compute the flat index from multi-dimensional indices
    size_t compute_flat_index(const std::vector<size_t> &indices) const
    {
        if (indices.size() != dims.size())
        {
            throw std::invalid_argument("Number of indices must match number of dimensions");
        }

        size_t index = 0;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (indices[i] >= dims[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * strides[i];
        }
        return index;
    }
};

#endif // MATRIX_HPP
