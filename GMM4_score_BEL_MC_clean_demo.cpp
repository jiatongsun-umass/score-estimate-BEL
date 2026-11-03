/*
  score_BEL_MC_clean_demo.cpp

  Estimate the full 3D score s(t,x,y,z) = grad log p(t,x,y,z) at (t,x,y,z) = (0.1,x,y,0) using BEL Monte Carlo,
  on a uniform grid in [-pi, pi) x [-pi, pi).
  Write one text file with columns: x y score_x score_y score_z.
  
  These are the full 3D score components evaluated at z = 0, not the score of a 2D marginal density.
*/

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace std;

// 1. Settings for parameters and constants. 
// Edit GRID_SIZE (grid size) and N_SAMPLE (number of independent trajectories) here.
constexpr int DIM = 3;
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double SIGMA = 1.5;                // noise magnitude in SDE
constexpr double TIME = 0.1;
constexpr int TIME_STEPS = 100;
constexpr double DT = TIME / TIME_STEPS;
constexpr int GRID_SIZE = 50;                // 50 x 50 query points
constexpr int N_SAMPLES = 20000;             // number of independent Monte Carlo trajectories per grid point
constexpr uint64_t SEED = 20261030;
constexpr const char* OUTPUT_FILE = "gmm4_abc_bel_score_t01_z0.txt";


// 2. GMM4 initial density
// Four Gaussian components: component k has covariance STDDEV[k]^2 * I.
constexpr array<double, 4> WEIGHTS{0.35, 0.25, 0.25, 0.15};
constexpr array<array<double, DIM>, 4> MEANS{{
    {-1.60, -1.20,  0.50},
    { 1.40, -1.90, -1.10},
    { 0.30,  1.70,  1.80},
    {-2.40,  2.20, -2.30}
}};
constexpr array<double, 4> STDDEV{0.45, 0.60, 0.35, 0.70};

// Wrap any real-valued coordinate to the torus interval [-pi, pi).
double wrap_pi(double x) 
{
    x = fmod(x + PI, 2.0 * PI); 
    if (x < 0.0)  // fmod may return a negative remainder
        x += 2.0 * PI;
    return x - PI;
}


// Wrapped 1D Gaussian density on the torus.
double wrapped_normal(double difference, double sd) 
{
    const double u = remainder(difference, 2.0 * PI);  // Reduce the coordinate difference (x - mean) to one fundamental period.
    const double factor = 1.0 / (sqrt(2.0 * PI) * sd);
    double density = 0.0;

    // p_wrap(x) = sum_{k in Z} N(x + 2*pi*k; 0, sd^2). 
    // We approximate the infinite sum using k = -3, ..., 3, which is sufficiently accurate for the GMM4 standard deviations.
    for (int k = -3; k <= 3; k++)                        
    {
        const double v = u + 2.0 * PI * k;
        density += factor * exp(-0.5 * v * v / (sd * sd));
    }
    return density;
}

// Evaluate the initial density p_0 of the four-component Gaussian mixture model (GMM4) on the 3D torus.
double initial_density(const array<double, DIM>& position) 
{
    double density = 0.0;
    for (int k = 0; k < 4; k++) 
    {
        double term = WEIGHTS[k];
        for (int i = 0; i < DIM; i++) // Compute the 3D wrapped Gaussian density as a product of independent 1D wrapped Gaussian densities.
        {
            term *= wrapped_normal(position[i] - MEANS[k][i], STDDEV[k]);
        }
        density += term;
    }
    return density;
}


// 3. Feynman-Kac auxiliary process Y(t), Jacobian process J = dY / dY(0), and BEL integral
/*
  Advance one auxiliary trajectory by one time step DT.
  The Feynman-Kac auxiliary process for the divergence-free ABC flow is
      dY = b(Y) dt + SIGMA dW. (b(Y) = -f(Y))
  To compute the BEL estimator, we simultaneously evolve
      J = dY / dY_0,        dJ/dt = -Df(Y) J,
      S = integral J^T dW.
  Numerical updates:
      Y_new = wrap(Y + b(Y)*DT + SIGMA*dW),
      J_new = (I + A*DT + 0.5*(A*DT)^2) J,
      S_new = S + J^T*dW,
  where b(Y) = -f(Y), A = Db(Y) = -Df(Y), and
      dW = sqrt(DT) * gaussian.
*/
void advance_path(array<double, DIM>& position,       // position Y = (Y_1, Y_2, Y_3)
                  array<array<double, DIM>, DIM>& J,  // Jacobian process J = dY / dY(0)
                  array<double, DIM>& integral,
                  const array<double, DIM>& gaussian) 
{
    // Feynman-Kac auxiliary drift b = -f and its derivative matrix A = Db, at the current position.
    array<double, DIM> drift{};
    array<array<double, DIM>, DIM> A{};
    for (int i = 0; i < DIM; i++) 
    {
        // Cyclic indexing for the three-dimensional ABC flow.
        const int previous = (i + DIM - 1) % DIM;
        const int next = (i + 1) % DIM;
        drift[i] = -(sin(position[previous]) + cos(position[next])); 
        A[i][previous] = -cos(position[previous]);
        A[i][next] = sin(position[next]);
    }


    // Left-endpoint Ito sum: use J before updating it.
    // The same Brownian increment drives the position and the BEL integral.
    const double sqrt_dt = sqrt(DT);
    for (int j = 0; j < DIM; j++) 
    {
        double increment = 0.0;
        for (int k = 0; k < DIM; k++) 
        {
            increment += J[k][j] * (sqrt_dt * gaussian[k]);
        }
        integral[j] += increment;
    }


    // Compute the Jacobian process using the second-order Taylor approximation: J_new = (I + A*DT + 0.5*(A*DT)^2) J.
    array<array<double, DIM>, DIM> first{};   // (A*DT)*J
    array<array<double, DIM>, DIM> second{};  // (A*DT)^2*J
    // First-order matrix contribution.
    for (int i = 0; i < DIM; i++) 
    {
        for (int j = 0; j < DIM; j++) 
        {
            for (int k = 0; k < DIM; k++) 
            {
                first[i][j] += (A[i][k] * DT) * J[k][j];
            }
        }
    }

    // Second-order matrix contribution.
    for (int i = 0; i < DIM; i++) 
    {
        for (int j = 0; j < DIM; j++) 
        {
            for (int k = 0; k < DIM; k++) 
            {
                second[i][j] += (A[i][k] * DT) * first[k][j];
            }
        }
    }

    // Combine the Taylor terms to obtain the updated Jacobian.
    for (int i = 0; i < DIM; i++) 
    {
        for (int j = 0; j < DIM; j++) 
        {
            J[i][j] += first[i][j] + 0.5 * second[i][j];
        }
    }

    // Euler-Maruyama step for the auxiliary position Y(t), followed by wrapping.
    for (int i = 0; i < DIM; i++) 
    {
        position[i] = wrap_pi(position[i] + drift[i] * DT + SIGMA * sqrt_dt * gaussian[i]);
    }
}



// 4. BEL Monte Carlo score estimator
/*
  Estimate the pointwise score s(t,x) = grad log p(t,x)
  using the BEL Monte Carlo formula.
  For the divergence-free ABC flow, the density satisfies
      p(t,x) = E[p0(Y_t)],
  where the auxiliary process is
      dY = -f(Y) dt + SIGMA dW,    Y_0 = x.
  The BEL formula gives
      grad p(t,x) = E[p0(Y_t) * S_t] / (SIGMA * t),
  where
      S_t = integral_0^t J_s^T dW_s,
  and J_s is the Jacobian of the auxiliary stochastic flow.
  Therefore,
      score(t,x) = grad p(t,x) / p(t,x).

  We approximate these expectations using independent Monte Carlo trajectories and take their ratio.
  Input:
      point : spatial point x where the score is evaluated.
      seed  : random seed for reproducibility.
  Output:
      The three components of the estimated score at (TIME, point).
*/
array<double, DIM> estimate_score(const array<double, DIM>& point,uint64_t seed) 
{
    mt19937_64 generator(seed);
    normal_distribution<double> normal(0.0, 1.0);

    // Accumulate the density and gradient Monte Carlo contributions.
    double density_sum = 0.0;
    array<double, DIM> gradient_sum{};

    for (int sample = 0; sample < N_SAMPLES; sample++) 
    {
        // Each trajectory starts at the query point, with J=I and zero Ito integral.
        array<double, DIM> position = point;
        array<array<double, DIM>, DIM> J{};
        array<double, DIM> integral{};
        for (int i = 0; i < DIM; i++)
        {
            J[i][i] = 1.0;
        }

        // Simulate one trajectory using independent Brownian increments.
        for (int step = 0; step < TIME_STEPS; step++)
        {
            array<double, DIM> gaussian{};
            for (int j = 0; j < DIM; j++)
            {
                gaussian[j] = normal(generator);
            }
            advance_path(position, J, integral, gaussian);
        }

        // Accumulate density and BEL gradient contributions.
        const double p = initial_density(position);
        density_sum += p;
        for (int j = 0; j < DIM; j++)
        {
            gradient_sum[j] += p * integral[j] / (SIGMA * TIME);
        }
    }

    if (!(density_sum > 0.0) || !std::isfinite(density_sum)) 
    {
        throw runtime_error("BEL density estimate is nonpositive or nonfinite");
    }

    // Take ONE ratio after summing all trajectories. The sample-count factors cancel.
    array<double, DIM> score{};
    for (int j = 0; j < DIM; j++) 
    {
        score[j] = gradient_sum[j] / density_sum;
        if (!isfinite(score[j])) 
        {
            throw runtime_error("BEL score estimate is nonfinite");
        }
    }
    return score;
}



int main() 
{
    static_assert(TIME_STEPS > 0 && GRID_SIZE > 0 && N_SAMPLES > 0, "Simulation parameters must be positive");
    static_assert(TIME > 0.0 && SIGMA > 0.0, "Time and noise strength must be positive");

    ofstream output(OUTPUT_FILE);
    if (!output) 
    {
        cerr << "Cannot open output file: " << OUTPUT_FILE << '\n';
        return 1;
    }
    output << setprecision(17); // Preserve double-precision values in the output.
    output << "# BEL score on t=" << TIME
           << ", z=0; x,y in [-pi,pi)\n";
    output << "# grid_size=" << GRID_SIZE << " mc_samples=" << N_SAMPLES
           << " time_steps=" << TIME_STEPS << " dt=" << DT
           << " sigma=" << SIGMA << " seed=" << SEED << '\n';
    output << "# x y score_x score_y score_z\n";

    try 
    {
        // Uniform grid on [-pi, pi)^2 at z=0.
        // Exclude +pi because it represents the same torus point as -pi.
        for (int ix = 0; ix < GRID_SIZE; ix++) 
        {
            const double x = -PI + 2.0 * PI * ix / GRID_SIZE; 
            for (int iy = 0; iy < GRID_SIZE; iy++) 
            {
                const double y = -PI + 2.0 * PI * iy / GRID_SIZE; 
                const array<double, DIM> point{x, y, 0.0}; 
                const uint64_t point_id = static_cast<uint64_t>(ix) * GRID_SIZE + iy;    // Assign a reproducible seed to each grid point.
                const array<double, DIM> score = estimate_score(point, SEED + point_id);
                output << x << ' ' << y << ' ' << score[0] << ' '
                       << score[1] << ' ' << score[2] << '\n';
            }
        }
    } 
    catch (const exception& error) 
    {
        cerr << "Computation failed: " << error.what() << '\n';
        return 1;
    }

    output.close();  // flush buffered output and check for write errors
    if (!output) 
    {
        cerr << "Cannot finish writing output file: " << OUTPUT_FILE << '\n';
        return 1;
    }
    cout << "Saved " << GRID_SIZE * GRID_SIZE << " pointwise scores to " << OUTPUT_FILE << '\n';
    return 0;
}