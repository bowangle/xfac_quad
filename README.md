# xfac_quad

Quad-precision fork of [xfac](https://github.com/your-org/xfac).

Adds `float128` and `dd_128` support alongside `double`.  
Tests are split per type to reduce compilation memory.

The same project provides grids in `include/QTgrid-quad/`, numerical headers in
`include/xfac_quad/`, and high-level fitting headers in `include/xfac_quad_runner/`.
Their tests live in the corresponding directories under `test/`.

## Build

```bash
bash ./compile.sh      # builds all three suites with the root CMakeLists.txt
```

## Run tests

```bash
./run_test.sh                   # all three suites
./run_test.sh -slow             # all suites, including core benchmarks
./run_test_qtgrid.sh            # grids only
./run_test_xfac_quad.sh         # numerical core only (-slow also supported)
./run_test_xfac_quad_runner.sh  # runner only
```
