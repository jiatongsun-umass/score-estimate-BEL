# GMM4 Score Estimation with BEL Monte Carlo

This repo contains a C++ implementation of the Bismut–Elworthy–Li (BEL) Monte Carlo estimator for pointwise score estimation. The demo considers a four-component Gaussian mixture (GMM4) initial density evolving under the three-dimensional ABC flow with additive noise on a periodic domain. It estimates the score (the gradient of the log-density) on the slice `t = 0.1, z = 0` and saves the results to a text file.

## Dependencies

The following are needed to compile and run this demo.

- A C++17 compiler, such as GCC or Clang.
- The C++ standard library. No external numerical libraries are required.

## Running the experiments

From the directory containing `GMM4_score_BEL_MC_clean_demo.cpp`, compile and run:

```bash
g++ -std=c++17 -O2 GMM4_score_BEL_MC_clean_demo.cpp -o bel_demo
./bel_demo
```

For Clang, replace `g++` with `clang++`. The program runs sequentially and takes no command-line arguments.

The simulation settings are defined near the top of the source file:

- `GRID_SIZE = 50`. This gives a uniform grid of 2,500 points in `[-pi, pi) × [-pi, pi)` at `z = 0`.
- `N_SAMPLES = 20000`. This is the number of independent Monte Carlo trajectories per grid point.
- `TIME = 0.1` and `TIME_STEPS = 100`. These give a time step of `0.001`.
- `SIGMA = 1.5`. This is the noise coefficient.
- `SEED = 20261030`. This is the base random seed.

For example, for a quick trial, set `GRID_SIZE = 5` and `N_SAMPLES = 1000`, then recompile and run. Reducing the sample count generally produces noisier estimates. The default settings require a longer serial computation.

## Output

The results are saved to `gmm4_abc_bel_score_t01_z0.txt` in the current working directory. An existing file with the same name is overwritten.

Lines beginning with `#` contain the simulation settings and column names. Each data row contains:

```text
x y score_x score_y score_z
```

Here `score_x`, `score_y`, and `score_z` are the three components of the estimated score at `(t, x, y, z) = (0.1, x, y, 0)`. These are the full three-dimensional score components evaluated on the slice, not the score of a two-dimensional marginal density.

## References

For further theoretical and implementation details, please refer to our paper (arXiv link to be added).
