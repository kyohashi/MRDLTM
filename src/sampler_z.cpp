#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector sample_z_cpp(
    NumericVector u_cit,
    NumericVector eta_zct_flat,
    NumericVector beta_zi_flat,
    NumericMatrix x_it_matrix,
    IntegerVector cust_idx,
    IntegerVector item_idx,
    IntegerVector time_idx,
    int n_topic,
    int n_item,
    int n_cust,
    int n_var
) {
  int N = u_cit.size();
  IntegerVector z_new(N);

  // SWE Comment: Pre-allocate a buffer for probabilities to avoid
  // repeated memory allocation inside the loop, which is a common performance killer.
  NumericVector log_probs(n_topic);

  for (int n = 0; n < N; ++n) {
    // Convert R 1-based indices to C++ 0-based indices
    int c = cust_idx[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    double u = u_cit[n];

    double max_log_p = -1e308; // Initialize with a very small number for log-sum-exp trick

    for (int z = 0; z < n_topic; ++z) {
      // 1. Calculate the contribution of Topic Occupancy (eta)
      // Note: The baseline topic (usually the last one) has eta = 0.
      double eta_val = 0.0;
      if (z < n_topic - 1) {
        // Flattened index calculation for 3D array: [z, c, t]
        // Engineering Note: Ensure this matching logic matches R's array storage (Column-major)
        eta_val = eta_zct_flat[z + c * (n_topic - 1) + t * (n_topic - 1) * n_cust];
      }

      // 2. Calculate the contribution of Marketing Response (x * beta)
      double xb = 0.0;
      for (int v = 0; v < n_var; ++v) {
        // Accessing x_it_matrix: row = i + t*n_item, col = v
        // beta_zi_flat: flattened 3D array index [z, i, v]
        xb += x_it_matrix(i + t * n_item, v) * beta_zi_flat[z + i * n_topic + v * n_topic * n_item];
      }

      // 3. Compute Log-Posterior proportional to: log P(y|z) + log P(z)
      // Since error epsilon ~ N(0, 1), log-likelihood is -0.5 * (u - xb)^2
      double log_p = eta_val - 0.5 * std::pow(u - xb, 2);
      log_probs[z] = log_p;

      if (log_p > max_log_p) max_log_p = log_p;
    }

    // SWE Comment: Numeric Stability - The Log-Sum-Exp Trick.
    // Subtracting the maximum value prevents numerical overflow when calling std::exp().
    double sum_p = 0.0;
    for (int z = 0; z < n_topic; ++z) {
      log_probs[z] = std::exp(log_probs[z] - max_log_p);
      sum_p += log_probs[z];
    }

    // 4. Draw a new topic from the categorical distribution
    double r = R::runif(0, 1) * sum_p;
    double cumulative_p = 0.0;
    int sampled_z = n_topic;
    for (int z = 0; z < n_topic; ++z) {
      cumulative_p += log_probs[z];
      if (r <= cumulative_p) {
        sampled_z = z + 1; // Convert back to R's 1-based indexing
        break;
      }
    }
    z_new[n] = sampled_z;
  }

  return z_new;
}
