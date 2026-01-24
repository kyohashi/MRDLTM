#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
NumericVector sample_u_cpp(
    IntegerVector y_cit,
    NumericVector beta_flat,  // [Z, M, I]
    NumericVector x_flat,     // [M, T, I]
    IntegerVector z_cit,      // Assigned topics for each observation
    IntegerVector item_idx,
    IntegerVector time_idx,
    int n_topic, int n_item, int n_time, int n_cust, int n_var
) {
  int N = y_cit.size();
  NumericVector u_new(N);

  // Map flat vectors to Cubes (no copy)
  cube beta_cube(beta_flat.begin(), n_topic, n_var, n_item, false);
  cube x_cube(x_flat.begin(), n_var, n_time, n_item, false);

  // Step 1: Precompute XB lookup table [Z x T x I]
  // This avoids redundant inner loops for each observation N
  cube xb_table(n_topic, n_time, n_item);
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      // Calculate linear predictor for all topics at once for this (i, t)
      xb_table.slice(i).col(t) = beta_cube.slice(i) * x_cube.slice(i).col(t);
    }
  }

  // Step 2: Main Sampling loop
  // Using single thread to ensure thread-safety for R::pnorm/R::qnorm
  for (int n = 0; n < N; ++n) {
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    int z = z_cit[n] - 1; // Current assigned topic
    int y = y_cit[n];

    // Retrieve precomputed linear predictor
    double xb = xb_table(z, t, i);

    // Truncated Normal Sampling using Inverse Transform Method
    double p_lower, p_upper;
    if (y == 1) {
      p_lower = R::pnorm(0.0, xb, 1.0, 1, 0); // Phi(0)
      p_upper = 1.0;
    } else {
      p_lower = 0.0;
      p_upper = R::pnorm(0.0, xb, 1.0, 1, 0); // Phi(0)
    }

    // Thread-safe random draw using Armadillo
    double u_rand = arma::randu();
    double target_p = p_lower + u_rand * (p_upper - p_lower);

    // Stability guards
    if (target_p < 1e-10) target_p = 1e-10;
    if (target_p > 1.0 - 1e-10) target_p = 1.0 - 1e-10;

    // Inverse CDF to get u ~ TN(xb, 1) using R::qnorm
    u_new[n] = R::qnorm(target_p, xb, 1.0, 1, 0);
  }

  return u_new;
}
