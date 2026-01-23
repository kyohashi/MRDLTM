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
    NumericVector u_cit,
    NumericVector eta_flat,   // [Z-1, C, T]
    NumericVector beta_flat,  // [Z, M, I]
    NumericVector x_flat,     // [M, T, I]
    IntegerVector cust_idx,
    IntegerVector item_idx,
    IntegerVector time_idx,
    NumericVector rand_u,     // Pre-generated uniforms
    int n_topic, int n_item, int n_time, int n_cust, int n_var
) {
  int N = u_cit.size();
  IntegerVector z_new(N);

  cube eta_cube(eta_flat.begin(), n_topic - 1, n_cust, n_time, false);
  cube beta_cube(beta_flat.begin(), n_topic, n_var, n_item, false);
  cube x_cube(x_flat.begin(), n_var, n_time, n_item, false);

  // Precompute XB table [Z x T x I]
  cube xb_table(n_topic, n_time, n_item);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      xb_table.slice(i).col(t) = beta_cube.slice(i) * x_cube.slice(i).col(t);
    }
  }

#pragma omp parallel
{
  vec log_probs(n_topic);
#pragma omp for
  for (int n = 0; n < N; ++n) {
    int c = cust_idx[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    double u = u_cit[n];

    // --- Equation (8): Softmax Probability Calculation ---
    double sum_exp_eta = 1.0;
    for (int k = 0; k < n_topic - 1; ++k) {
      sum_exp_eta += std::exp(eta_cube(k, c, t));
    }

    double max_log_p = -1e308;
    for (int z = 0; z < n_topic; ++z) {
      double pi_z;
      if (z < n_topic - 1) {
        pi_z = std::exp(eta_cube(z, c, t)) / sum_exp_eta;
      } else {
        pi_z = 1.0 / sum_exp_eta; // Baseline topic Z
      }

      // Likelihood: u_cit ~ N(x_it' * beta_zi, 1)
      double log_lik = -0.5 * std::pow(u - xb_table(z, t, i), 2);
      log_probs[z] = std::log(pi_z + 1e-15) + log_lik;

      if (log_probs[z] > max_log_p) max_log_p = log_probs[z];
    }

    double sum_p = 0.0;
    for (int z = 0; z < n_topic; ++z) {
      log_probs[z] = std::exp(log_probs[z] - max_log_p);
      sum_p += log_probs[z];
    }

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
