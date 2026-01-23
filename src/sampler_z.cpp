#include <RcppArmadillo.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
IntegerVector sample_z_cpp(
    IntegerVector y_cit,
    NumericVector eta_flat,
    NumericVector beta_flat,
    NumericVector x_flat,
    IntegerVector cust_idx,
    IntegerVector item_idx,
    IntegerVector time_idx,
    NumericVector rand_u,
    int n_topic, int n_item, int n_time, int n_cust, int n_var
) {
  int N = y_cit.size();
  IntegerVector z_new(N);

  // Map to Cubes (no copy)
  cube eta_cube(eta_flat.begin(), n_topic - 1, n_cust, n_time, false);
  cube beta_cube(beta_flat.begin(), n_topic, n_var, n_item, false);
  cube x_cube(x_flat.begin(), n_var, n_time, n_item, false);

  // Step 1: Precompute XB lookup table [Z x T x I]
  cube xb_table(n_topic, n_time, n_item);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      xb_table.slice(i).col(t) = beta_cube.slice(i) * x_cube.slice(i).col(t);
    }
  }

  // Step 2: Sampling loop
#pragma omp parallel
{
  vec log_probs(n_topic);
#pragma omp for
  for (int n = 0; n < N; ++n) {
    int c = cust_idx[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    int y = y_cit[n];

    double max_log_p = -1e308;

    for (int z = 0; z < n_topic; ++z) {
      // --- Log-Prior ---
      // Baseline normalization constant is omitted as it cancels out during sampling.
      double log_pi_z = (z < n_topic - 1) ? eta_cube(z, c, t) : 0.0;

      // --- Log-Likelihood (Probit) ---
      double xb = xb_table(z, t, i);
      double log_omega;
      if (y == 1) {
        log_omega = R::pnorm(xb, 0.0, 1.0, 1, 1); // log(Phi(xb))
      } else {
        log_omega = R::pnorm(xb, 0.0, 1.0, 0, 1); // log(1 - Phi(xb))
      }

      log_probs[z] = log_pi_z + log_omega;
      if (log_probs[z] > max_log_p) max_log_p = log_probs[z];
    }

    // Step 3: Convert to relative weights using Log-Sum-Exp trick
    double sum_p = 0.0;
    for (int z = 0; z < n_topic; ++z) {
      log_probs[z] = std::exp(log_probs[z] - max_log_p);
      sum_p += log_probs[z];
    }

    // Step 4: Categorical sampling via cumulative sum
    double r = rand_u[n] * sum_p;
    double cumulative_p = 0.0;
    int sampled_z = n_topic;
    for (int z = 0; z < n_topic; ++z) {
      cumulative_p += log_probs[z];
      if (r <= cumulative_p) {
        sampled_z = z + 1;
        break;
      }
    }
    z_new[n] = sampled_z;
  }
}
return z_new;
}
