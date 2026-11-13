#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <complex>
#include <functional>
#include <fstream>
#include <string>
#include <limits>
#include <random>
#include <stdexcept>

#include <grid.h>
#include <xfac_quad/xfac_quad.hpp>
#include <xfac_quad/tree/tensor_tree.hpp>

namespace xfac_quad_runner {

// Linespace
template <typename Scalar>
std::vector<Scalar> linspace(Scalar a, Scalar b, int N)
{
    std::vector<Scalar> x;
    x.reserve(N);

    if (N == 1) {
        x.push_back(a);
        return x;
    }

    Scalar step = (b - a) / Scalar(N - 1);

    for (int i = 0; i < N; ++i) {
        x.push_back(a + Scalar(i) * step);
    }

    return x;
}

template <typename Scalar>
void compute_stats(
    const std::vector<Scalar>& v,
    Scalar& mean,
    Scalar& min,
    Scalar& max)
{
    using Real = Scalar;
    using std::abs;
    using boost::multiprecision::abs;

    Real sum = Real(0);

    Real min_abs = std::numeric_limits<Real>::infinity();
    Real max_abs = Real(0);

    int count = 0;

    for (const auto& x : v)
    {
        Real ax = abs(x);

        sum += ax;

        if (ax < min_abs) min_abs = ax;
        if (ax > max_abs) max_abs = ax;

        count++;
    }

    Real mean_abs = (count > 0)
        ? sum / Real(count)
        : std::numeric_limits<Real>::quiet_NaN();

    // cast back to Scalar (important if Scalar is complex)
    mean = Scalar(mean_abs);
    min  = Scalar(min_abs);
    max  = Scalar(max_abs);
}

// output type of error calculation
template <typename Scalar>
struct TTErrorOnGrid {
    using Complex = std::complex<Scalar>;

    std::vector<Scalar> real_abs;
    std::vector<Scalar> real_rel;
    std::vector<Scalar> imag_abs;
    std::vector<Scalar> imag_rel;
    std::vector<Scalar> abs_abs;
    std::vector<Scalar> abs_rel;

    std::vector<Scalar> l_point;
    std::vector<Complex> l_ref;
    std::vector<Complex> l_tt;

    Scalar real_abs_mean, real_abs_min, real_abs_max;
    Scalar real_rel_mean, real_rel_min, real_rel_max;

    Scalar imag_abs_mean, imag_abs_min, imag_abs_max;
    Scalar imag_rel_mean, imag_rel_min, imag_rel_max;

    Scalar abs_abs_mean, abs_abs_min, abs_abs_max;
    Scalar abs_rel_mean, abs_rel_min, abs_rel_max;

    void compute_summary()
    {
        compute_stats(real_abs, real_abs_mean, real_abs_min, real_abs_max);
        compute_stats(real_rel, real_rel_mean, real_rel_min, real_rel_max);

        compute_stats(imag_abs, imag_abs_mean, imag_abs_min, imag_abs_max);
        compute_stats(imag_rel, imag_rel_mean, imag_rel_min, imag_rel_max);

        compute_stats(abs_abs, abs_abs_mean, abs_abs_min, abs_abs_max);
        compute_stats(abs_rel, abs_rel_mean, abs_rel_min, abs_rel_max);
    }
};

template <typename Scalar>
void save_TTErrorOnGrid(const TTErrorOnGrid<Scalar>& e,
                        const std::string& filename)
{
    std::ofstream file(filename);

    if (!file)
        throw std::runtime_error("Cannot open file");

    file << std::setprecision(std::numeric_limits<Scalar>::max_digits10);

    using Complex = std::complex<Scalar>;

    auto write_vec = [&](const char* name, const std::vector<Scalar>& v)
    {
        file << name << " " << v.size() << "\n";
        for (const auto& x : v) file << x << " ";
        file << "\n\n";
    };

    auto write_cvec = [&](const char* name, const std::vector<Complex>& v)
    {
        file << name << " " << v.size() << "\n";
        for (const auto& x : v)
            file << "(" << x.real() << "," << x.imag() << ") ";
        file << "\n\n";
    };

    write_vec("l_point",  e.l_point);
    write_cvec("l_ref", e.l_ref);
    write_cvec("l_tt",  e.l_tt);
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const TTErrorOnGrid<Scalar>& e)
{
    auto print_stats = [&](const char* name,
                           Scalar mean,
                           Scalar min,
                           Scalar max)
    {
        os << name << ":\n"
           << "  mean = " << mean << "\n"
           << "  min  = " << min << "\n"
           << "  max  = " << max << "\n";
    };

    os << "================ TT ERROR REPORT ================\n";

    print_stats("Real abs error", e.real_abs_mean, e.real_abs_min, e.real_abs_max);
    print_stats("Real rel error", e.real_rel_mean, e.real_rel_min, e.real_rel_max);

    print_stats("Imag abs error", e.imag_abs_mean, e.imag_abs_min, e.imag_abs_max);
    print_stats("Imag rel error", e.imag_rel_mean, e.imag_rel_min, e.imag_rel_max);

    print_stats("Abs error", e.abs_abs_mean, e.abs_abs_min, e.abs_abs_max);
    print_stats("Abs rel error", e.abs_rel_mean, e.abs_rel_min, e.abs_rel_max);

    os << "=================================================\n";

    return os;
}

// Error calculation:
template <typename Scalar, typename Sint>
TTErrorOnGrid<Scalar> error_TT_on_grid_point(
    const xfac_quad::TensorTrain<std::complex<Scalar>>& tt,
    std::function<std::complex<Scalar>(std::vector<int>)> gf_id,
    const QTGrid<Scalar, Sint>& grid,
    int nb_point)
{
    using Complex = std::complex<Scalar>;

    TTErrorOnGrid<Scalar> out;
    out.real_abs.resize(nb_point);
    out.real_rel.resize(nb_point);
    out.imag_abs.resize(nb_point);
    out.imag_rel.resize(nb_point);
    out.abs_abs.resize(nb_point);
    out.abs_rel.resize(nb_point);

    std::vector<Scalar> l_point = linspace(grid.get_a(), grid.get_b(), nb_point);
    std::vector<Complex> l_ref_i(nb_point);
    std::vector<Complex> l_res_tt(nb_point);


    for (int i = 0; i < nb_point; i++)
    {
        Scalar x = l_point[i];
        std::vector<int> id = grid.coord_to_id(x);

        Complex ref_i = gf_id(id);
        l_ref_i[i] = ref_i;

        Complex res_tt = tt.eval(id);
        l_res_tt[i] = res_tt;

        Complex diff = ref_i - res_tt;

        Scalar abs_diff = std::abs(diff);
        Scalar ref_norm = std::abs(ref_i);

        // signed errors
        out.real_abs[i] = diff.real();
        out.imag_abs[i] = diff.imag();
        out.abs_abs[i]  = abs_diff;

        // relative
        if (ref_norm != Scalar(0))
        {
            out.real_rel[i] = diff.real() / ref_norm;
            out.imag_rel[i] = diff.imag() / ref_norm;
            out.abs_rel[i]  = abs_diff / ref_norm;
        }
        else
        {
            out.real_rel[i] = 0;
            out.imag_rel[i] = 0;
            out.abs_rel[i]  = 0;
        }
    }
    out.l_point = l_point;
    out.l_ref = l_ref_i;
    out.l_tt = l_res_tt;
    out.compute_summary();
    return out;
}

// Tree/ND counterpart of TTErrorOnGrid. Each point stores all physical coordinates.
template <typename Scalar>
struct TTreeErrorOnGrid {
    using Complex = std::complex<Scalar>;

    std::vector<Scalar> real_abs;
    std::vector<Scalar> real_rel;
    std::vector<Scalar> imag_abs;
    std::vector<Scalar> imag_rel;
    std::vector<Scalar> abs_abs;
    std::vector<Scalar> abs_rel;

    std::vector<std::vector<Scalar>> l_point;
    std::vector<Complex> l_ref;
    std::vector<Complex> l_tt;

    Scalar real_abs_mean, real_abs_min, real_abs_max;
    Scalar real_rel_mean, real_rel_min, real_rel_max;
    Scalar imag_abs_mean, imag_abs_min, imag_abs_max;
    Scalar imag_rel_mean, imag_rel_min, imag_rel_max;
    Scalar abs_abs_mean, abs_abs_min, abs_abs_max;
    Scalar abs_rel_mean, abs_rel_min, abs_rel_max;

    void compute_summary()
    {
        compute_stats(real_abs, real_abs_mean, real_abs_min, real_abs_max);
        compute_stats(real_rel, real_rel_mean, real_rel_min, real_rel_max);
        compute_stats(imag_abs, imag_abs_mean, imag_abs_min, imag_abs_max);
        compute_stats(imag_rel, imag_rel_mean, imag_rel_min, imag_rel_max);
        compute_stats(abs_abs, abs_abs_mean, abs_abs_min, abs_abs_max);
        compute_stats(abs_rel, abs_rel_mean, abs_rel_min, abs_rel_max);
    }
};

template <typename Scalar>
void save_TTreeErrorOnGrid(const TTreeErrorOnGrid<Scalar>& e,
                          const std::string& filename)
{
    std::ofstream file(filename);
    if (!file)
        throw std::runtime_error("save_TTreeErrorOnGrid: cannot open file");
    file << std::scientific
         << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
    const size_t dim = e.l_point.empty() ? 0 : e.l_point.front().size();
    file << "l_point " << e.l_point.size() << ' ' << dim << "\n";
    for (const auto& point : e.l_point) {
        for (const Scalar& coordinate : point) file << coordinate << ' ';
        file << "\n";
    }
    file << "\nl_ref " << e.l_ref.size() << "\n";
    for (const auto& value : e.l_ref)
        file << '(' << value.real() << ',' << value.imag() << ") ";
    file << "\n\nl_tt " << e.l_tt.size() << "\n";
    for (const auto& value : e.l_tt)
        file << '(' << value.real() << ',' << value.imag() << ") ";
    file << "\n";
    if (!file)
        throw std::runtime_error("save_TTreeErrorOnGrid: failed to write file");
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const TTreeErrorOnGrid<Scalar>& e)
{
    auto print_stats = [&](const char* name, Scalar mean, Scalar min, Scalar max)
    {
        os << name << ":\n"
           << "  mean = " << mean << "\n"
           << "  min  = " << min << "\n"
           << "  max  = " << max << "\n";
    };
    os << "============== TREE ERROR REPORT ================\n";
    print_stats("Real abs error", e.real_abs_mean, e.real_abs_min, e.real_abs_max);
    print_stats("Real rel error", e.real_rel_mean, e.real_rel_min, e.real_rel_max);
    print_stats("Imag abs error", e.imag_abs_mean, e.imag_abs_min, e.imag_abs_max);
    print_stats("Imag rel error", e.imag_rel_mean, e.imag_rel_min, e.imag_rel_max);
    print_stats("Abs error", e.abs_abs_mean, e.abs_abs_min, e.abs_abs_max);
    print_stats("Abs rel error", e.abs_rel_mean, e.abs_rel_min, e.abs_rel_max);
    os << "=================================================\n";
    return os;
}

/// Test nb_point/2 Halton points and the remaining uniformly random grid points.
/// Both sets cover the full hypercube. The first points are Halton (index 1 onward),
/// followed by random points with seed 42; odd counts give the extra point to random.
/// Coordinates recorded in the report are the actual quantized grid coordinates.
template <typename Scalar, typename Sint>
TTreeErrorOnGrid<typename Eigen::NumTraits<Scalar>::Real> error_TTree_on_grid_point(
    const xfac_quad::TensorTree<Scalar>& tt,
    std::function<Scalar(std::vector<int>)> gf_id,
    const MultQTGrid<typename Eigen::NumTraits<Scalar>::Real, Sint>& grid,
    int nb_point)
{
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    using Complex = std::complex<Real>;
    if (nb_point <= 0)
        throw std::invalid_argument("error_TTree_on_grid_point: nb_point must be positive");

    TTreeErrorOnGrid<Real> out;
    out.real_abs.resize(nb_point);
    out.real_rel.resize(nb_point);
    out.imag_abs.resize(nb_point);
    out.imag_rel.resize(nb_point);
    out.abs_abs.resize(nb_point);
    out.abs_rel.resize(nb_point);
    out.l_point.reserve(nb_point);
    out.l_ref.resize(nb_point);
    out.l_tt.resize(nb_point);

    // Generate one distinct prime base per dimension, without a fixed dimension limit.
    std::vector<int> bases;
    bases.reserve(grid.get_dim());
    for (int candidate = 2; bases.size() < static_cast<size_t>(grid.get_dim()); candidate++) {
        bool prime = true;
        for (int divisor : bases) {
            if (divisor > candidate / divisor) break;
            if (candidate % divisor == 0) {
                prime = false;
                break;
            }
        }
        if (prime) bases.push_back(candidate);
    }
    auto radical_inverse = [](int index, int base) {
        Real value = Real(0);
        Real factor = Real(1);
        while (index > 0) {
            factor /= Real(base);
            value += factor * Real(index % base);
            index /= base;
        }
        return value;
    };

    const int nb_halton = nb_point / 2;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> bit(0, 1);
    for (int i = 0; i < nb_point; i++) {
        std::vector<int> id;
        if (i < nb_halton) {
            std::vector<Real> point(grid.get_dim());
            for (int axis = 0; axis < grid.get_dim(); axis++) {
                const Real u = radical_inverse(i + 1, bases[axis]);
                point[axis] = grid.get_a()[axis]
                    + u * (grid.get_b()[axis] - grid.get_a()[axis]);
            }
            id = grid.coord_to_id(point);
        } else {
            // Independent uniform bits sample the whole grid, including Sint > 64 bits.
            id.resize(grid.get_tensorLen());
            for (int& value : id) value = bit(rng);
        }
        out.l_point.push_back(grid.id_to_coord(id));
        const Complex reference(gf_id(id));
        const Complex approximation(tt.eval(id));
        const Complex diff = reference - approximation;
        using std::abs;
        const Real ref_norm = abs(reference);
        out.l_ref[i] = reference;
        out.l_tt[i] = approximation;
        out.real_abs[i] = diff.real();
        out.imag_abs[i] = diff.imag();
        out.abs_abs[i] = abs(diff);
        if (ref_norm != Real(0)) {
            out.real_rel[i] = diff.real() / ref_norm;
            out.imag_rel[i] = diff.imag() / ref_norm;
            out.abs_rel[i] = out.abs_abs[i] / ref_norm;
        } else {
            out.real_rel[i] = Real(0);
            out.imag_rel[i] = Real(0);
            out.abs_rel[i] = Real(0);
        }
    }
    out.compute_summary();
    return out;
}

} // namespace xfac_quad_runner
