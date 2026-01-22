#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
IntegerVector sample_z_cpp(
    NumericVector u_cit,
    NumericVector eta_flat,
    NumericVector beta_flat,
    NumericVector x_flat,
    IntegerVector cust_idx,
    IntegerVector item_idx,
    IntegerVector time_idx,
    NumericVector rand_u,
    int n_topic, int n_item, int n_time, int n_cust, int n_var
) {
  int N = u_cit.size();
  IntegerVector z_new(N);

  cube eta_cube(eta_flat.begin(), n_topic - 1, n_cust, n_time, false);
  cube beta_cube(beta_flat.begin(), n_topic, n_var, n_item, false);
  cube x_cube(x_flat.begin(), n_var, n_time, n_item, false);

  // --- NEW: Precompute XB table [Z x T x I] ---
  // This avoids redundant matrix-vector products for each customer
  cube xb_table(n_topic, n_time, n_item);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      xb_table.slice(i).col(t) = beta_cube.slice(i) * x_cube.slice(i).col(t);
    }
  }

#pragma omp parallel
{
  vec thread_log_probs(n_topic);
#pragma omp for
  for (int n = 0; n < N; ++n) {
    int c = cust_idx[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    double u = u_cit[n];

    double max_log_p = -1e308;
    for (int z = 0; z < n_topic; ++z) {
      double eta_val = (z < n_topic - 1) ? eta_cube(z, c, t) : 0.0;
      // Lookup precomputed XB
      double log_p = eta_val - 0.5 * std::pow(u - xb_table(z, t, i), 2);
      thread_log_probs[z] = log_p;
      if (log_p > max_log_p) max_log_p = log_p;
    }

    double sum_p = 0.0;
    for (int z = 0; z < n_topic; ++z) {
      thread_log_probs[z] = std::exp(thread_log_probs[z] - max_log_p);
      sum_p += thread_log_probs[z];
    }

    double r = rand_u[n] * sum_p;
    double cumulative_p = 0.0;
    int sampled_z = n_topic;
    for (int z = 0; z < n_topic; ++z) {
      cumulative_p += thread_log_probs[z];
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
