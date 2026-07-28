# xfac_quad

Quad-precision fork of [xfac](https://github.com/your-org/xfac).

Adds `float128` and `dd_128` support alongside `double`.  
Tests are split per type to reduce compilation memory.

## Build

```bash
mkdir build && cd build
cmake .. && make -j
```

## Run tests

```bash
./run_test.sh          # CI + SVD tests (all types)
./run_test.sh -slow    # also run benchmarks
```
