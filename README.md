# xfac_quad

Quad-precision fork of [xfac](https://github.com/your-org/xfac).

Adds `float128` and `dd_128` support alongside `double`.  
Tests are split per type to reduce compilation memory.

The same project provides numerical headers in `include/xfac_quad/` and
high-level fitting headers in `include/xfac_quad_runner/`. Their tests live in
`test/xfac_quad/` and `test/xfac_quad_runner/`, respectively.

## Build

```bash
bash ./compile.sh      # builds both suites with the root CMakeLists.txt
```

## Run tests

```bash
./run_test.sh                   # both suites
./run_test.sh -slow             # both suites, including core benchmarks
./run_test_xfac_quad.sh         # numerical core only (-slow also supported)
./run_test_xfac_quad_runner.sh  # runner only
```
