#pragma once
#include "type_double_double.h"
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <complex>
#include <type_traits>

namespace xfac_quad {

using std::vector;

// ============================================================
// Tensor3D<T> — replacement for arma::Cube<T>
// ============================================================
template<typename T>
struct Tensor3D {
    using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Stride_dim = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

    Eigen::Index n_rows, n_cols, n_slices;  // matching Armadillo naming
    std::vector<T> data;

    Tensor3D() : n_rows(0), n_cols(0), n_slices(0) {}
    Tensor3D(Eigen::Index r, Eigen::Index c, Eigen::Index s)
        : n_rows(r), n_cols(c), n_slices(s), data(r * c * s) {}
    /// Construct from raw data (copies)
    Tensor3D(T const* ptr, size_t r, size_t c, size_t s)
        : n_rows(static_cast<Eigen::Index>(r))
        , n_cols(static_cast<Eigen::Index>(c))
        , n_slices(static_cast<Eigen::Index>(s))
        , data(ptr, ptr + r*c*s) {}

    [[nodiscard]] inline Eigen::Index idx(Eigen::Index i, Eigen::Index j, Eigen::Index k) const {
        return k * n_rows * n_cols + j * n_rows + i;
    }

    T& operator()(Eigen::Index i, Eigen::Index j, Eigen::Index k) {
        return data[idx(i, j, k)];
    }
    const T& operator()(Eigen::Index i, Eigen::Index j, Eigen::Index k) const {
        return data[idx(i, j, k)];
    }

    // ---- slice views (matching Armadillo's .slice(k)) ----
    Eigen::Map<MatrixX> slice(Eigen::Index k) {
        return Eigen::Map<MatrixX>(data.data() + k * n_rows * n_cols, n_rows, n_cols);
    }
    Eigen::Map<const MatrixX> slice_const(Eigen::Index k) const {
        return Eigen::Map<const MatrixX>(data.data() + k * n_rows * n_cols, n_rows, n_cols);
    }

    // ---- col(j) = column as matrix (Armadillo: .col(j)) ----
    Eigen::Map<MatrixX, 0, Stride_dim> col_as_mat(Eigen::Index j) {
        return Eigen::Map<MatrixX, 0, Stride_dim>(
            data.data() + j * n_rows, n_rows, n_slices,
            Stride_dim(n_rows * n_cols, 1));
    }
    Eigen::Map<const MatrixX, 0, Stride_dim> col_as_mat_const(Eigen::Index j) const {
        return Eigen::Map<const MatrixX, 0, Stride_dim>(
            data.data() + j * n_rows, n_rows, n_slices,
            Stride_dim(n_rows * n_cols, 1));
    }

    // ---- row_as_mat(i) (Armadillo) ----
    Eigen::Map<MatrixX, 0, Stride_dim> row_as_mat(Eigen::Index i) {
        return Eigen::Map<MatrixX, 0, Stride_dim>(
            data.data() + i, n_cols, n_slices,
            Stride_dim(n_rows * n_cols, n_rows));
    }
    Eigen::Map<const MatrixX, 0, Stride_dim> row_as_mat_const(Eigen::Index i) const {
        return Eigen::Map<const MatrixX, 0, Stride_dim>(
            data.data() + i, n_cols, n_slices,
            Stride_dim(n_rows * n_cols, n_rows));
    }
};

// ============================================================
// Helper functions matching xfac's cubemat_helper.h
// ============================================================

/// reshape a cube as a matrix B(i,jk)=A(i,j,k)
template<class T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
cube_as_matrix1(Tensor3D<T>& A) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        A.data.data(), A.n_rows, A.n_cols * A.n_slices);
}

template<class T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
cube_as_matrix1(Tensor3D<T> const& A) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        A.data.data(), A.n_rows, A.n_cols * A.n_slices);
}

/// reshape a cube as a matrix B(ij,k)=A(i,j,k)
template<class T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
cube_as_matrix2(Tensor3D<T>& A) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        A.data.data(), A.n_rows * A.n_cols, A.n_slices);
}

template<class T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
cube_as_matrix2(Tensor3D<T> const& A) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        A.data.data(), A.n_rows * A.n_cols, A.n_slices);
}

// ============================================================
// Save / Load (Armadillo-compatible format)
// ============================================================
template<typename RealT>
RealT parse_real(const std::string& s) {
    RealT val;
    std::istringstream iss(s);
    iss >> val;
    if (iss.fail()) throw std::runtime_error("Failed to parse: " + s);
    return val;
}

template<typename T>
std::vector<Tensor3D<T>> load_vector_tensor(std::istream& in) {
    using RealT = typename T::value_type;
    Eigen::Index nBit;
    in >> nBit;
    std::vector<Tensor3D<T>> Xs;
    Xs.reserve(nBit);
    for (Eigen::Index t = 0; t < nBit; ++t) {
        std::string header;
        in >> header;
        if (header != "ARMA_CUB_TXT_FC016")
            throw std::runtime_error("Bad header");
        Eigen::Index r, c, s;
        in >> r >> c >> s;
        Tensor3D<T> X(r, c, s);
        for (Eigen::Index k = 0; k < X.n_slices; ++k)
            for (Eigen::Index i = 0; i < X.n_rows; ++i)
                for (Eigen::Index j = 0; j < X.n_cols; ++j) {
                    std::string token;
                    in >> token;
                    if (token.size() < 5) throw std::runtime_error("Bad complex token");
                    if (token.front() == '(') token.erase(token.begin());
                    if (token.back() == ')') token.pop_back();
                    auto comma = token.find(',');
                    if (comma == std::string::npos) throw std::runtime_error("Missing comma");
                    RealT re = parse_real<RealT>(token.substr(0, comma));
                    RealT im = parse_real<RealT>(token.substr(comma + 1));
                    X(i, j, k) = T(re, im);
                }
        Xs.push_back(std::move(X));
    }

    if constexpr (std::is_same_v<RealT, dd_128>) {
        // New dd_128 files append an exact trailer after the unchanged legacy
        // decimal representation. Old files simply reach EOF here. Loaders
        // instantiated for another RealT return before examining the trailer.
        std::string marker;
        if (in >> marker && marker == "XFQD_DD128_EXACT_V1") {
            std::size_t exact_core_count;
            if (!(in >> exact_core_count) || exact_core_count != Xs.size()) {
                throw std::runtime_error("Invalid dd_128 exact TT core count");
            }

            for (std::size_t t = 0; t < Xs.size(); ++t) {
                Eigen::Index r, c, s;
                if (!(in >> r >> c >> s)) {
                    throw std::runtime_error("Invalid dd_128 exact TT dimensions");
                }

                Tensor3D<T>& X = Xs[t];
                if (r != X.n_rows || c != X.n_cols || s != X.n_slices) {
                    throw std::runtime_error("dd_128 exact TT dimensions do not match legacy data");
                }

                for (Eigen::Index k = 0; k < X.n_slices; ++k)
                    for (Eigen::Index i = 0; i < X.n_rows; ++i)
                        for (Eigen::Index j = 0; j < X.n_cols; ++j) {
                            double re_hi, re_lo, im_hi, im_lo;
                            if (!(in >> re_hi >> re_lo >> im_hi >> im_lo)) {
                                throw std::runtime_error("Invalid dd_128 exact TT value");
                            }

                            RealT re(dd_real(re_hi, re_lo));
                            RealT im(dd_real(im_hi, im_lo));
                            X(i, j, k) = T(re, im);
                        }
            }
        }
    }

    return Xs;
}

template<typename T>
std::vector<Tensor3D<T>> load_vector_tensor(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);
    return load_vector_tensor<T>(file);
}

template<class T>
void save_Tensor3D_to_arma(std::ostream& out, const std::vector<Tensor3D<T>>& Xs) {
    using RealT = typename T::value_type;
    out << Xs.size() << "\n";
    for (std::size_t t = 0; t < Xs.size(); ++t) {
        out << "ARMA_CUB_TXT_FC016\n";
        const Tensor3D<T>& X = Xs[t];
        out << X.n_rows << " " << X.n_cols << " " << X.n_slices << "\n";
        out << std::scientific;
        out << std::setprecision(std::numeric_limits<RealT>::digits10 + 5);
        for (Eigen::Index k = 0; k < X.n_slices; ++k) {
            for (Eigen::Index i = 0; i < X.n_rows; ++i) {
                for (Eigen::Index j = 0; j < X.n_cols; ++j) {
                    const auto& z = X(i, j, k);
                    out << "(" << z.real() << "," << z.imag() << ") ";
                }
                out << "\n";
            }
        }
        out << "\n";
    }

    if constexpr (std::is_same_v<RealT, dd_128>) {
        // Dual representation: preserve the legacy decimal TT above, then add
        // the exact two-double representation for new dd_128-aware loaders.
        out << "XFQD_DD128_EXACT_V1\n";
        out << Xs.size() << "\n";
        out << std::scientific
            << std::setprecision(std::numeric_limits<double>::max_digits10);

        for (const Tensor3D<T>& X : Xs) {
            out << X.n_rows << " " << X.n_cols << " " << X.n_slices << "\n";
            for (Eigen::Index k = 0; k < X.n_slices; ++k) {
                for (Eigen::Index i = 0; i < X.n_rows; ++i) {
                    for (Eigen::Index j = 0; j < X.n_cols; ++j) {
                        const auto& z = X(i, j, k);
                        const RealT re = z.real();
                        const RealT im = z.imag();
                        out << re.x[0] << " " << re.x[1] << " "
                            << im.x[0] << " " << im.x[1] << " ";
                    }
                    out << "\n";
                }
            }
        }
    }
}

template<typename T>
void save_Tensor3D_to_arma(std::vector<Tensor3D<T>> cores, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file");
    save_Tensor3D_to_arma(file, cores);
}

} // namespace xfac_quad
