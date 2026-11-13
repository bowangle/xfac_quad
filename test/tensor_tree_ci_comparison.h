#pragma once

#include <catch2/catch.hpp>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree_ci.hpp"

namespace tree_ci_comparison {

enum class Function { radial_exponential, gaussian, polynomial, cosines };

inline const char* name(Function f) {
    switch (f) {
    case Function::radial_exponential: return "exp(-r)";
    case Function::gaussian: return "gaussian";
    case Function::polynomial: return "complex_polynomial";
    case Function::cosines: return "cosines";
    }
    throw std::logic_error("Unknown comparison function");
}

inline std::uint64_t unsigned_setting(const char* variable, std::uint64_t fallback) {
    const char* text = std::getenv(variable);
    if (!text) return fallback;
    const std::string value(text);
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
        throw std::invalid_argument(std::string(variable) + " must be an unsigned integer");
    return parsed;
}

inline std::vector<int> choices(const char* variable, std::vector<int> allowed) {
    if (!std::getenv(variable)) return allowed;
    const auto selected = unsigned_setting(variable, 0);
    for (int value : allowed)
        if (selected == static_cast<std::uint64_t>(value)) return {value};
    throw std::invalid_argument(std::string("Unsupported value for ") + variable);
}

/// MultQTGrid stores each coordinate's bits contiguously, least-significant bit first.
/// Seed before either interpolation; integer draws do not depend on scalar precision.
inline std::vector<std::vector<int>> sample_indices(int dim, int bits, std::size_t count,
                                                     std::mt19937_64& rng) {
    std::uniform_int_distribution<std::uint64_t> draw(0, (std::uint64_t(1) << bits) - 1);
    std::vector<std::vector<int>> samples(count, std::vector<int>(dim * bits));
    for (auto& index : samples)
        for (int axis = 0; axis < dim; ++axis) {
            const auto k = draw(rng);
            for (int bit = 0; bit < bits; ++bit)
                index[axis * bits + bit] = static_cast<int>((k >> bit) & 1);
        }
    return samples;
}

/// Relabel makeTuckerTree's interleaved physical IDs (axis + bit*dim) into
/// grouped IDs (axis*bits + bit). Edges and neighbor/leg ordering are unchanged.
inline xfac_quad::TopologyTree grouped_tucker_tree(int dim, int bits) {
    const auto source = xfac_quad::makeTuckerTree(dim, bits);
    const int physical_count = dim * bits;
    const auto grouped_id = [=](int node) {
        return node < physical_count ? (node % dim) * bits + node / dim : node;
    };
    xfac_quad::TopologyTree result(grouped_id(source.root));
    for (int node : source.nodes) result.nodes.insert(grouped_id(node));
    for (const auto& [node, neighbors] : source.neigh) {
        auto& target = result.neigh[grouped_id(node)];
        for (int neighbor : neighbors.from_int()) target.push_back(grouped_id(neighbor));
    }
    return result;
}

template<typename Real, typename Scalar>
Scalar exact_function(Function kind, std::vector<Real> const& x) {
    using std::cos;
    using std::exp;
    using std::sqrt;
    Real radius_squared = Real(0), sum = Real(0);
    for (const Real& coordinate : x) {
        radius_squared += coordinate * coordinate;
        sum += coordinate;
    }
    switch (kind) {
    case Function::radial_exponential:
        return Scalar(exp(-sqrt(radius_squared)));
    case Function::gaussian:
        return Scalar(exp(-radius_squared));
    case Function::polynomial:
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
            const Scalar a(Real(1), Real(1));
            const Scalar b(Real(1)/Real(3), -Real(2)/Real(3));
            const Scalar c(-Real(2)/Real(5), Real(1)/Real(5));
            return a + b * sum * sum + c * sum * sum * sum;
        } else {
            throw std::logic_error("The polynomial requires complex scalars");
        }
    case Function::cosines: {
        Real weighted_sum = Real(0), cosine_sum = Real(0), cosine_product = Real(1);
        for (std::size_t j = 0; j < x.size(); ++j) {
            weighted_sum += Real(j + 1) * x[j];
            cosine_sum += cos(Real(2) * x[j]);
            cosine_product *= cos(x[j]);
        }
        return Scalar(cos(weighted_sum) + cosine_sum/Real(2) + cosine_product/Real(3));
    }
    }
    throw std::logic_error("Unknown comparison function");
}

inline std::string quoted(std::string const& value) {
    std::string result = "\"";
    for (char c : value) {
        if (c == '"') result += '"';
        result += c;
    }
    return result + '"';
}

template<typename Real>
struct Result {
    int iterations = 0;
    Real max_error = Real(0);
    std::size_t failed_points = 0, nonfinite_points = 0, worst_point = 0;
    std::size_t function_evaluations = 0, max_bond = 0;
    double training_seconds = 0, evaluation_seconds = 0, total_seconds = 0;
    std::string exception;
};

template<typename T>
std::size_t max_bond(xfac_quad::TensorCI2<T> const& ci) {
    std::size_t result = 0;
    for (const auto& indices : ci.Iset) result = std::max(result, indices.size());
    return result;
}

template<typename T>
std::size_t max_bond(xfac_quad::TensorTreeCI<T> const& ci) {
    std::size_t result = 0;
    for (const auto& entry : ci.Iset) result = std::max(result, entry.second.size());
    return result;
}

template<typename Real, typename Scalar, typename Build>
Result<Real> measure(Build build, std::vector<std::vector<int>> const& initial_pivots,
                     std::vector<std::vector<int>> const& samples,
                     std::vector<Scalar> const& exact, Real const& tolerance, int iterations = 10) {
    using std::abs;
    Result<Real> result;
    result.iterations = iterations;
    const auto start = std::chrono::steady_clock::now();
    auto evaluation_start = start;
    bool trained = false;
    try {
        auto ci = build();
        ci.addPivotsAllBonds(initial_pivots);
        ci.iterate(iterations);
        evaluation_start = std::chrono::steady_clock::now();
        trained = true;
        result.training_seconds = std::chrono::duration<double>(evaluation_start - start).count();
        result.function_evaluations = ci.f.nEval();
        result.max_bond = max_bond(ci);
        for (std::size_t p = 0; p < samples.size(); ++p) {
            const Real error = abs(ci.tt.eval(samples[p]) - exact[p]);
            if (!(error <= tolerance)) ++result.failed_points; // NaNs also fail
            if (!(error == error) || error > std::numeric_limits<Real>::max()) {
                ++result.nonfinite_points;
                result.worst_point = p;
            } else if (error > result.max_error) {
                result.max_error = error;
                result.worst_point = p;
            }
        }
        result.evaluation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - evaluation_start).count();
    } catch (std::exception const& error) {
        const auto failed = std::chrono::steady_clock::now();
        if (trained)
            result.evaluation_seconds = std::chrono::duration<double>(failed - evaluation_start).count();
        else
            result.training_seconds = std::chrono::duration<double>(failed - start).count();
        result.exception = error.what();
    }
    result.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

template<typename Real>
void report(const char* precision, Function kind, int a, int b, int dim, int bits,
            bool full_piv, std::size_t points, Real const& reltol, Real const& tolerance,
            const char* method, Result<Real> const& result, std::vector<int> const& worst_id) {
    // One file per precision; truncate once at the start of this executable's run.
    static std::ofstream file([&] {
        const char* path = std::getenv("XFAC_QUAD_TTCI_REPORT");
        return path ? std::string(path) : "tensor_tree_ci_compare_" + std::string(precision) + ".csv";
    }());
    if (!file) throw std::runtime_error("Cannot open TensorTreeCI comparison report");
    static bool header = false;
    if (!header) {
        file << "precision,function,a,b,dimension,bits,pivoting,points,seed,initial_pivots,reltol,absolute_tolerance,method,iterations,max_abs_error,failed_points,nonfinite_points,function_evaluations,max_bond,training_seconds,evaluation_seconds,total_seconds,worst_grid_id,status,exception\n";
        header = true;
    }
    std::string id;
    for (int bit : worst_id) id += bit ? '1' : '0';
    const bool passed = result.exception.empty() && result.failed_points == 0;
    file << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10)
         << precision << ',' << name(kind) << ',' << a << ',' << b << ',' << dim << ',' << bits << ','
         << (full_piv ? "full" : "rook") << ',' << points << ",42,5," << reltol << ',' << tolerance << ','
         << method << ',' << result.iterations << ',' << result.max_error << ',' << result.failed_points << ',' << result.nonfinite_points << ','
         << result.function_evaluations << ',' << result.max_bond << ',' << result.training_seconds << ','
         << result.evaluation_seconds << ',' << result.total_seconds << ',' << quoted(id) << ',' << (passed ? "PASS" : "FAIL") << ',' << quoted(result.exception) << std::endl;
    if (!file) throw std::runtime_error("Cannot write TensorTreeCI comparison report");
    std::cout << precision << ' ' << name(kind) << " [" << a << ',' << b << "] d=" << dim
              << " bits=" << bits << ' ' << (full_piv ? "full" : "rook") << ' ' << method
              << " max_abs_error=" << result.max_error << " failed=" << result.failed_points << '/' << points
              << " status=" << (passed ? "PASS" : "FAIL") << std::endl;
}

template<typename Real, typename Scalar>
void run(Function kind, const char* precision) {
    const auto dimensions = choices("XFAC_QUAD_TTCI_DIM", {1, 2, 3});
    const auto resolutions = choices("XFAC_QUAD_TTCI_BITS", {10, 30});
    const int dim = GENERATE_COPY(Catch::Generators::from_range(dimensions));
    const int bits = GENERATE_COPY(Catch::Generators::from_range(resolutions));
    const bool full_piv = false;
    const int a = -9, b = 5;
    const auto count = unsigned_setting("XFAC_QUAD_TTCI_POINTS", 1000);
    if (count == 0 || count > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("XFAC_QUAD_TTCI_POINTS must fit a positive size_t");
    const auto nb_point_to_test = static_cast<std::size_t>(count);
    const Real reltol = Real(10000) * std::numeric_limits<Real>::epsilon();
    const Real tolerance = Real(100000) * std::numeric_limits<Real>::epsilon();
    CAPTURE(precision, a, b, dim, bits, full_piv, nb_point_to_test, reltol, tolerance);

    const MultQTGrid<Real, std::int64_t> grid(std::vector<Real>(dim, Real(a)),
        std::vector<Real>(dim, Real(b)), std::vector<int>(dim, bits));
    // Recreate the same five distinct global pivots for both methods and every precision.
    // Continue the stream for evaluation points, rather than reusing the pivots as its prefix.
    std::mt19937_64 rng(42);
    std::vector<std::vector<int>> initial_pivots;
    while (initial_pivots.size() < 5) {
        auto candidate = sample_indices(dim, bits, 1, rng).front();
        if (std::find(initial_pivots.begin(), initial_pivots.end(), candidate) == initial_pivots.end())
            initial_pivots.push_back(std::move(candidate));
    }
    const auto samples = sample_indices(dim, bits, nb_point_to_test, rng);
    std::vector<Scalar> exact;
    exact.reserve(samples.size());
    for (const auto& index : samples)
        exact.push_back(exact_function<Real, Scalar>(kind, grid.id_to_coord(index)));
    const std::function<Scalar(std::vector<int>)> function = [&](std::vector<int> index) {
        return exact_function<Real, Scalar>(kind, grid.id_to_coord(index));
    };
    const std::vector<int> tensor_dimensions(grid.get_tensorLen(), 2);
    const auto tree = grouped_tucker_tree(dim, bits);
    REQUIRE(xfac_quad::isValid(tree));

    std::cout << "START " << precision << ' ' << name(kind) << " [" << a << ',' << b << "] d=" << dim
              << " bits=" << bits << ' ' << (full_piv ? "full" : "rook") << std::endl;
    const auto train = measure<Real, Scalar>([&] {
        xfac_quad::TensorCI2Param<Scalar> param;
        param.bondDim = 100;
        param.reltol = reltol;
        param.fullPiv = full_piv;
        return xfac_quad::TensorCI2<Scalar>(function, tensor_dimensions, param);
    }, initial_pivots, samples, exact, tolerance);
    report(precision, kind, a, b, dim, bits, full_piv, nb_point_to_test, reltol, tolerance,
           "TensorCI2", train, samples.at(train.worst_point));

    const auto branched = measure<Real, Scalar>([&] {
        xfac_quad::TensorTreeCIParam<Scalar> param;
        param.bondDim = 100;
        param.reltol = reltol;
        param.fullPiv = full_piv;
        return xfac_quad::TensorTreeCI<Scalar>(function, tree, tensor_dimensions, param);
    }, initial_pivots, samples, exact, tolerance, 5);
    report(precision, kind, a, b, dim, bits, full_piv, nb_point_to_test, reltol, tolerance,
           "TensorTreeCI", branched, samples.at(branched.worst_point));

    // Do not relax or scale the requested absolute threshold, including for large polynomials.
    for (const auto* result : {&train, &branched}) {
        INFO("method=" << (result == &train ? "TensorCI2" : "TensorTreeCI"));
        CAPTURE(result->exception, result->max_error, result->failed_points, result->worst_point);
        CHECK(result->exception.empty());
        if (result->exception.empty()) {
            CHECK(result->nonfinite_points == 0);
            CHECK(result->failed_points == 0);
            CHECK(result->max_error <= tolerance);
            CHECK(result->max_bond <= 100);
        }
    }
}

} // namespace tree_ci_comparison

// Each translation unit selects its real precision to keep compilation memory bounded.
TEST_CASE("MultQTGrid radial exponential comparison - " XFAC_QUAD_COMPARISON_NAME,
          "[.][slow][ttci_compare][radial_exponential]") {
    tree_ci_comparison::run<XFAC_QUAD_COMPARISON_REAL, XFAC_QUAD_COMPARISON_REAL>(
        tree_ci_comparison::Function::radial_exponential, XFAC_QUAD_COMPARISON_NAME);
}
TEST_CASE("MultQTGrid Gaussian comparison - " XFAC_QUAD_COMPARISON_NAME,
          "[.][slow][ttci_compare][gaussian]") {
    tree_ci_comparison::run<XFAC_QUAD_COMPARISON_REAL, XFAC_QUAD_COMPARISON_REAL>(
        tree_ci_comparison::Function::gaussian, XFAC_QUAD_COMPARISON_NAME);
}
TEST_CASE("MultQTGrid complex polynomial comparison - " XFAC_QUAD_COMPARISON_NAME,
          "[.][slow][ttci_compare][complex_polynomial]") {
    tree_ci_comparison::run<XFAC_QUAD_COMPARISON_REAL, std::complex<XFAC_QUAD_COMPARISON_REAL>>(
        tree_ci_comparison::Function::polynomial, XFAC_QUAD_COMPARISON_NAME);
}
TEST_CASE("MultQTGrid cosine comparison - " XFAC_QUAD_COMPARISON_NAME,
          "[.][slow][ttci_compare][cosines]") {
    tree_ci_comparison::run<XFAC_QUAD_COMPARISON_REAL, XFAC_QUAD_COMPARISON_REAL>(
        tree_ci_comparison::Function::cosines, XFAC_QUAD_COMPARISON_NAME);
}
