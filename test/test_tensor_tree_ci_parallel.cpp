#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <omp.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree_ci.hpp"

namespace {

int setting(const char* variable, int fallback) {
    const char* text = std::getenv(variable);
    if (!text) return fallback;
    const std::string value(text);
    int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 1)
        throw std::invalid_argument(std::string(variable) + " must be a positive integer");
    return parsed;
}

struct OpenMPSettings {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    OpenMPSettings() { omp_set_dynamic(0); }
    ~OpenMPSettings() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

std::vector<int> random_index(std::mt19937_64& rng) {
    std::uniform_int_distribution<std::uint64_t> draw(0, (std::uint64_t(1) << 30) - 1);
    const auto index = draw(rng);
    std::vector<int> bits(30);
    for (int bit = 0; bit < 30; ++bit) bits[bit] = static_cast<int>((index >> bit) & 1);
    return bits;
}

std::ofstream& csv() {
    static std::ofstream file([] {
        const char* path = std::getenv("XFAC_QUAD_TTCI_PARALLEL_REPORT");
        return path ? std::string(path) : std::string("tensor_tree_ci_parallel.csv");
    }());
    if (!file) throw std::runtime_error("Cannot open parallel CI benchmark CSV");
    static bool header = false;
    if (!header) {
        file << "precision,method,a,b,dimension,bits,threads,actual_threads,repetitions,seed,initial_pivots,evaluation_delay_us,reltol,absolute_tolerance,training_median_seconds,training_min_seconds,training_max_seconds,speedup_vs_1_thread,max_abs_error,function_evaluations,max_bond,status\n";
        header = true;
    }
    return file;
}

template<typename T>
std::size_t max_bond(xfac_quad::TensorCI2<T> const& ci) {
    std::size_t rank = 0;
    for (const auto& indices : ci.Iset) rank = std::max(rank, indices.size());
    return rank;
}

template<typename T>
std::size_t max_bond(xfac_quad::TensorTreeCI<T> const& ci) {
    std::size_t rank = 0;
    for (const auto& entry : ci.Iset) rank = std::max(rank, entry.second.size());
    return rank;
}

template<typename Real>
struct Measurement {
    double seconds;
    Real max_error;
    std::size_t evaluations, rank;
    bool accurate;
};

template<typename Real, typename Build>
Measurement<Real> measure(Build build, std::vector<std::vector<int>> const& pivots,
                          std::vector<std::vector<int>> const& samples,
                          std::vector<Real> const& exact, Real const& tolerance) {
    using std::abs;
    // Adaptive rook pivoting uses rand(); reset before every fresh reconstruction.
    std::srand(42);
    const auto start = std::chrono::steady_clock::now();
    auto ci = build();
    ci.addPivotsAllBonds(pivots);
    ci.iterate(10);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // Keep validation and destruction outside the timed training region.
    Measurement<Real> result{seconds, Real(0), ci.f.nEval(), max_bond(ci), true};
    for (std::size_t p = 0; p < samples.size(); ++p) {
        const Real error = abs(ci.tt.eval(samples[p]) - exact[p]);
        if (!(error <= tolerance)) result.accurate = false;
        if (error > result.max_error) result.max_error = error;
    }
    return result;
}

template<typename Real>
void benchmark(const char* precision) {
    using namespace xfac_quad;
    OpenMPSettings restore;
    const int available = std::min(omp_get_num_procs(), omp_get_thread_limit());
    const int maximum_threads = std::min(available, setting("XFAC_QUAD_TTCI_MAX_THREADS", 8));
    const int repetitions = setting("XFAC_QUAD_TTCI_TIMING_REPEATS", 5);
    std::vector<int> thread_counts{1};
    for (int count = 2; count <= maximum_threads; count *= 2) thread_counts.push_back(count);
    if (thread_counts.back() != maximum_threads) thread_counts.push_back(maximum_threads);
    INFO("available OpenMP processors=" << available);
    REQUIRE(available >= 1);

    const MultQTGrid<Real, std::int64_t> grid({Real(-9)}, {Real(5)}, {30});
    const auto exact_function = [&](std::vector<int> const& bits) -> Real {
        using std::abs;
        using std::exp;
        return Real(exp(-abs(grid.id_to_coord(bits)[0])));
    };
    const std::function<Real(std::vector<int>)> function = [&](std::vector<int> bits) -> Real {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        return exact_function(bits);
    };
    std::mt19937_64 rng(42);
    std::vector<std::vector<int>> pivots;
    while (pivots.size() < 5) {
        auto index = random_index(rng);
        if (std::find(pivots.begin(), pivots.end(), index) == pivots.end()) pivots.push_back(std::move(index));
    }
    std::vector<std::vector<int>> samples;
    std::vector<Real> exact;
    for (int p = 0; p < 1000; ++p) {
        samples.push_back(random_index(rng));
        exact.push_back(exact_function(samples.back()));
    }
    const Real reltol = Real(10000) * std::numeric_limits<Real>::epsilon();
    const Real tolerance = Real(100000) * std::numeric_limits<Real>::epsilon();
    const std::vector<int> dimensions(30, 2);
    const auto tree = makeTuckerTree(1, 30);
    const auto train_builder = [&] {
        TensorCI2Param<Real> param;
        param.bondDim = 100;
        param.reltol = reltol;
        param.fullPiv = false;
        return TensorCI2<Real>(function, dimensions, param);
    };
    const auto tree_builder = [&] {
        TensorTreeCIParam<Real> param;
        param.bondDim = 100;
        param.reltol = reltol;
        param.fullPiv = false;
        return TensorTreeCI<Real>(function, tree, dimensions, param);
    };
    std::array<double, 2> baseline{};
    std::array<std::size_t, 2> baseline_evaluations{}, baseline_rank{};
    const char* names[] = {"TensorCI2", "TensorTreeCI"};

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        int actual_threads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        CAPTURE(precision, threads, actual_threads, repetitions);
        REQUIRE(actual_threads == threads);
        std::cout << "START " << precision << " exp(-r), 1D, 30 bits, threads=" << actual_threads << std::endl;

        // Warm up both paths, including the OpenMP team. Every measurement still constructs
        // a new CI and therefore starts with an empty function cache and the same pivot seed.
        const auto train_warmup = measure<Real>(train_builder, pivots, samples, exact, tolerance);
        const auto tree_warmup = measure<Real>(tree_builder, pivots, samples, exact, tolerance);
        CHECK(train_warmup.accurate);
        CHECK(tree_warmup.accurate);
        std::array<std::vector<Measurement<Real>>, 2> runs;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            // Alternate algorithm order to reduce systematic first/second measurement bias.
            for (int order = 0; order < 2; ++order) {
                const int method = (order + repetition) % 2;
                if (method == 0) runs[0].push_back(measure<Real>(train_builder, pivots, samples, exact, tolerance));
                else runs[1].push_back(measure<Real>(tree_builder, pivots, samples, exact, tolerance));
            }
        }
        for (int method = 0; method < 2; ++method) {
            CAPTURE(names[method]);
            std::vector<double> times;
            Real max_error = Real(0);
            bool accurate = true;
            for (const auto& run : runs[method]) {
                times.push_back(run.seconds);
                if (run.max_error > max_error) max_error = run.max_error;
                accurate = accurate && run.accurate;
                CHECK(run.accurate);
                CHECK(run.rank <= 100);
                CHECK(run.evaluations == runs[method].front().evaluations);
                CHECK(run.rank == runs[method].front().rank);
            }
            std::sort(times.begin(), times.end());
            const double median = times.size() % 2 ? times[times.size()/2]
                : (times[times.size()/2 - 1] + times[times.size()/2])/2;
            if (threads == 1) {
                baseline[method] = median;
                baseline_evaluations[method] = runs[method].front().evaluations;
                baseline_rank[method] = runs[method].front().rank;
            }
            CHECK(runs[method].front().evaluations == baseline_evaluations[method]);
            CHECK(runs[method].front().rank == baseline_rank[method]);
            const double speedup = baseline[method] / median;
            auto& out = csv();
            out << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10)
                << precision << ',' << names[method] << ",-9,5,1,30," << threads << ',' << actual_threads << ','
                << repetitions << ",42,5,500," << reltol << ',' << tolerance << ',' << median << ','
                << times.front() << ',' << times.back() << ',' << speedup << ',' << max_error << ','
                << runs[method].front().evaluations << ',' << runs[method].front().rank << ','
                << (accurate ? "PASS" : "FAIL") << std::endl;
            REQUIRE(out.good());
            std::cout << precision << ' ' << names[method] << " threads=" << threads
                      << " median_seconds=" << median << " speedup=" << speedup
                      << " max_abs_error=" << max_error << std::endl;
        }
    }
}

} // namespace

TEST_CASE("30-bit 1D exp(-r) OpenMP scaling - double", "[.][slow][ttci_parallel]") {
    benchmark<double>("double");
}
TEST_CASE("30-bit 1D exp(-r) OpenMP scaling - float128", "[.][slow][ttci_parallel]") {
    benchmark<float128>("float128");
}
TEST_CASE("30-bit 1D exp(-r) OpenMP scaling - dd128", "[.][slow][ttci_parallel]") {
    benchmark<dd_128>("dd128");
}
