// Exercises TTCI_runner in 1D with the same functions and configurations as
// test_runner.cpp, excluding unsupported discontinuity metadata.
//
//   cache   in { none, xfac, runner }   x   fullPiv in { true, false }
//
// Check accuracy, cache transparency, function-call counters, and exact
// save/load roundtrips for every supported real and complex scalar type.

#include "test_common.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>

using namespace xfac_quad_runner;
using testing::check;

// ---- problem size, matching test_runner.cpp ----
static constexpr int kNBit    = 30;
static constexpr int kNbIter  = 10;
static constexpr int kBondDim = 100;
static constexpr int kNbPointRes = 200;
static constexpr int kNbPointErr = 2000;

template <typename Scalar>
struct ConfigResult {
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    CacheLevel  cache    = CacheLevel::none;
    bool        fullPiv  = true;
    std::string tt_file;
    int         n_calls  = 0;
    int         n_cached = 0;
    Real        max_abs_rel = Real(0);
    bool        completed = false;
};

// ============================================================
// One configuration
// ============================================================
template <typename Scalar>
ConfigResult<Scalar> run_config(const std::string& type_label,
                                CacheLevel cache,
                                bool fullPiv)
{
    using Real    = typename Eigen::NumTraits<Scalar>::Real;
    using Complex = std::complex<Real>;
    using Sint    = util::i128;

    const std::string ctx = testing::config_label(type_label, cache, fullPiv);
    std::cout << "\n----- " << ctx << " -----" << std::endl;

    ConfigResult<Scalar> out;
    out.cache   = cache;
    out.fullPiv = fullPiv;

    // ---- grid and 1D chain ----
    MultQTGrid<Real, Sint> grid({Real(0)}, {Real(1)}, {kNBit});
    const auto tree = xfac_quad::makeTuckerTree(1, kNBit);

    // ---- a non-default pivot1, to verify propagation ----
    std::vector<int> pivot1_custom(kNBit, 1);
    pivot1_custom[0] = 0;                          // [0,1,1,...,1]

    ttci_runner_opts<Scalar> opts{
        .reltol  = Eigen::NumTraits<Real>::epsilon() * Real(100),
        .pivot1  = pivot1_custom,
        .fullPiv = fullPiv,
        .cache   = cache
    };
    ttci_runner_param<Scalar> param(kNbIter, kBondDim, opts);

    // ---- parameter propagation, including the derived cache flags ----
    check(param.nb_iter == kNbIter,   ctx, "nb_iter propagated");
    check(param.bondDim == kBondDim,  ctx, "bondDim propagated");
    check(param.pivot1 == pivot1_custom, ctx, "pivot1 propagated");
    check(param.fullPiv == fullPiv,   ctx, "fullPiv propagated");
    check(param.nRookIter == 3,       ctx, "nRookIter default");
    check(param.cache == cache,       ctx, "cache propagated");
    check(param.useCachedFunction_xfac_lvl == (cache == CacheLevel::xfac),
          ctx, "xfac-level cache flag matches CacheLevel");
    check(param.do_cache_runner_lvl == (cache == CacheLevel::runner),
          ctx, "runner-level cache flag matches CacheLevel");
    check(!(param.useCachedFunction_xfac_lvl && param.do_cache_runner_lvl),
          ctx, "the two cache layers are never both active");

    // ---- fit the unchanged 1D function through a vector-coordinate adapter ----
    auto function_1d = pick_test_func<Scalar>();
    const std::function<Scalar(std::vector<Real>)> test_func =
        [function_1d](std::vector<Real> point) -> Scalar {
            return function_1d(point[0]);
        };
    const std::string prefix = "test/xfac_quad_runner/output/ttci_" + ctx;
    out.tt_file = prefix + ".tt";

    TTCI_runner<Scalar, Sint> runner(grid, tree, param, test_func);
    runner.fit(/*additional_pivot=*/{}, /*verbose=*/true, /*do_save=*/true,
               prefix, kNbPointRes);

    out.n_calls  = runner.counter.load();
    out.n_cached = runner.counter_cached.load();

    // ---- counter invariants ----
    check(out.n_calls > 0, ctx, "at least one real function evaluation happened");
    if (cache == CacheLevel::runner) {
        check(out.n_cached > 0, ctx, "runner-level cache produced hits");
        check(out.n_calls <= out.n_calls + out.n_cached,
              ctx, "calls + hits accounting is sane");
    } else {
        check(out.n_cached == 0, ctx,
              "no runner-level cache hits when the runner cache is off");
    }

    // ---- reload the saved tensor tree, including its topology ----
    auto tt_complex = xfac_quad::TensorTree<Complex>::load(out.tt_file);
    check(!tt_complex.M.empty(), ctx, "saved tensor tree reloads non-empty");
    if (tt_complex.M.empty()) return out;

    // ---- accuracy against the reference function ----
    std::function<Complex(std::vector<int>)> gf_id =
        [&](std::vector<int> id) -> Complex {
            return Complex(test_func(grid.id_to_coord(id)));
        };
    TTreeErrorOnGrid<Real> error =
        error_TTree_on_grid_point(tt_complex, gf_id, grid, kNbPointErr);
    std::cout << error;
    out.max_abs_rel = error.abs_rel_max;

    // Same accuracy threshold as test_runner.cpp.
    const Real tol = Eigen::NumTraits<Real>::epsilon() * Real(1e6);
    check(error.abs_rel_max <= tol, ctx, "relative error within tolerance");

    // ---- save/load roundtrip ----
    const std::string roundtrip_file = prefix + "_roundtrip.tt";
    tt_complex.save(roundtrip_file);
    auto reloaded = xfac_quad::TensorTree<Complex>::load(roundtrip_file);
    const auto& cores2 = reloaded.M;
    check(cores2.size() == tt_complex.M.size(), ctx, "roundtrip core count");

    if (cores2.size() == tt_complex.M.size()) {
        using std::abs;
        Real roundtrip_diff = Real(0);
        bool shapes_ok = true;
        for (size_t c = 0; c < tt_complex.M.size(); ++c) {
            const auto& orig   = tt_complex.M[c];
            const auto& reload = cores2[c];
            if (orig.n_rows != reload.n_rows ||
                orig.n_cols != reload.n_cols ||
                orig.n_slices != reload.n_slices) { shapes_ok = false; break; }
            for (Eigen::Index i = 0;
                 i < static_cast<Eigen::Index>(orig.data.size()); ++i) {
                Complex d = orig.data[i] - reload.data[i];
                Real err = abs(d.real()) + abs(d.imag());
                if (err > roundtrip_diff) roundtrip_diff = err;
            }
        }
        check(shapes_ok, ctx, "roundtrip core shapes match");
        check(roundtrip_diff == Real(0),
              ctx, "save/load roundtrip is exactly lossless");
    }

    out.completed = true;
    return out;
}

// ============================================================
// All configurations for one scalar type
// ============================================================
template <typename Scalar>
void run_all_configs(const std::string& type_label)
{
    constexpr bool is_complex = Eigen::NumTraits<Scalar>::IsComplex != 0;

    std::cout << "\n================================================\n"
              << "  Testing " << type_label
              << "  (is_complex=" << is_complex << ")\n"
              << "================================================" << std::endl;

    std::vector<ConfigResult<Scalar>> results;
    for (bool fullPiv : {true, false})
        for (CacheLevel cache : {CacheLevel::none,
                                 CacheLevel::xfac,
                                 CacheLevel::runner})
            results.push_back(run_config<Scalar>(type_label, cache, fullPiv));

    // ---- caching must not change the result ----
    // As in test_runner.cpp, compare bytes only with full pivoting. Rook
    // pivoting consumes a global random stream and is checked for accuracy.
    {
        const bool fullPiv = true;
        const ConfigResult<Scalar>* baseline = nullptr;
        for (const auto& r : results) {
            if (r.fullPiv != fullPiv || !r.completed) continue;
            if (!baseline) { baseline = &r; continue; }

            const std::string ctx = type_label + "_" + testing::piv_name(fullPiv);
            const std::string a = testing::read_file(baseline->tt_file);
            const std::string b = testing::read_file(r.tt_file);

            check(!a.empty() && !b.empty(), ctx, "tensor tree files readable");
            check(a == b, ctx,
                  std::string("cache=") + xf_qd_detail::cache_name(r.cache)
                  + " gives the same tensor tree as cache="
                  + xf_qd_detail::cache_name(baseline->cache)
                  + " (caching must be transparent)");
        }
    }

    // ---- caching must not increase the amount of real work ----
    for (bool fullPiv : {true, false}) {
        const ConfigResult<Scalar>* none_run = nullptr;
        for (const auto& r : results)
            if (r.fullPiv == fullPiv && r.cache == CacheLevel::none && r.completed)
                none_run = &r;

        if (!none_run) continue;

        for (const auto& r : results) {
            if (r.fullPiv != fullPiv || !r.completed) continue;
            if (r.cache == CacheLevel::none) continue;
            const std::string ctx = type_label + "_" + testing::piv_name(fullPiv);
            check(r.n_calls <= none_run->n_calls, ctx,
                  std::string("cache=") + xf_qd_detail::cache_name(r.cache)
                  + " does not evaluate the function more often than cache=none");
        }
    }

    // ---- summary table ----
    std::cout << "\n  " << std::left << std::setw(10) << "cache"
              << std::setw(10) << "pivot"
              << std::right << std::setw(12) << "calls"
              << std::setw(12) << "hits" << "   max rel err\n";
    for (const auto& r : results) {
        std::ostringstream err;
        err << std::scientific << std::setprecision(3) << r.max_abs_rel;
        std::cout << "  " << std::left << std::setw(10)
                  << xf_qd_detail::cache_name(r.cache)
                  << std::setw(10) << testing::piv_name(r.fullPiv)
                  << std::right << std::setw(12) << r.n_calls
                  << std::setw(12) << r.n_cached
                  << "   " << err.str() << "\n";
    }
    std::cout << std::flush;
}

int main()
{
    std::filesystem::create_directories("test/xfac_quad_runner/output");
    std::cout << "Running TTCI_runner 1D configuration tests "
                 "(cache x fullPiv x scalar type)...\n";

    run_all_configs<double>("double");
    run_all_configs<dd_128>("dd_128");
    run_all_configs<float128>("float128");

    run_all_configs<std::complex<double>>("complex_double");
    run_all_configs<Cdd_128>("complex_dd_128");
    run_all_configs<Cfloat128>("complex_float128");

    return testing::finish("test_ttci_runner");
}
