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
    IntegerVector y_cit,      // Observed binary outcomes (0/1)
    NumericVector eta_flat,   // Topic prevalence [Z-1, C, T]
    NumericVector beta_flat,  // Item coefficients [Z, M, I]
    NumericVector x_flat,     // Marketing variables [M, T, I]
    IntegerVector cust_idx,
    IntegerVector item_idx,
    IntegerVector time_idx,
    NumericVector rand_u,     // Pre-generated uniforms for sampling
    int n_topic, int n_item, int n_time, int n_cust, int n_var
) {
  int N = y_cit.size();
  IntegerVector z_new(N);

  // Map to Cubes (no copy)
  cube eta_cube(eta_flat.begin(), n_topic - 1, n_cust, n_time, false);
  cube beta_cube(beta_flat.begin(), n_topic, n_var, n_item, false);
  cube x_cube(x_flat.begin(), n_var, n_time, n_item, false);

  // Step 1: Precompute Log-Likelihood tables [Z x T x I]
  // Precomputing both log(Phi(xb)) and log(1 - Phi(xb)) to avoid R::pnorm in the main loop
  cube log_phi_y1(n_topic, n_time, n_item);
  cube log_phi_y0(n_topic, n_time, n_item);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      // Calculate linear predictor xb for each topic z
      vec xb_vec = beta_cube.slice(i) * x_cube.slice(i).col(t);
      for (int z = 0; z < n_topic; ++z) {
        double xb = xb_vec[z];
        // y=1: log(Phi(xb)), y=0: log(1-Phi(xb))
        log_phi_y1(z, t, i) = R::pnorm(xb, 0.0, 1.0, 1, 1);
        log_phi_y0(z, t, i) = R::pnorm(xb, 0.0, 1.0, 0, 1);
      }
    }
  }

  // Step 2: Main Sampling loop
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
      double log_pi_z = (z < n_topic - 1) ? eta_cube(z, c, t) : 0.0;

      // --- Log-Likelihood (Lookup from precomputed tables) ---
      // Replaced expensive R::pnorm calls with simple table lookups
      double log_omega = (y == 1) ? log_phi_y1(z, t, i) : log_phi_y0(z, t, i);

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

// [[Rcpp::export]]
List sample_z_with_prob_cpp(
    IntegerVector y_cit,      // Observed binary outcomes (0/1)
    NumericVector eta_flat,   // Topic prevalence [Z-1, C, T]
    NumericVector beta_flat,  // Item coefficients [Z, M, I]
    NumericVector x_flat,     // Marketing variables [M, T, I]
    IntegerVector cust_idx,
    IntegerVector item_idx,
    IntegerVector time_idx,
    NumericVector rand_u,     // Pre-generated uniforms for sampling
    IntegerVector prob_idx,   // 1-based indices of observations to return probabilities for
    int n_topic, int n_item, int n_time, int n_cust, int n_var
) {
  int N = y_cit.size();
  int n_prob = prob_idx.size();
  IntegerVector z_new(N);

  // Output matrix for probabilities [n_prob x n_topic]
  NumericMatrix prob_out(n_prob, n_topic);

  // Create lookup: obs_idx -> prob_out row (or -1 if not tracked)
  std::vector<int> prob_lookup(N, -1);
  for (int j = 0; j < n_prob; ++j) {
    int obs_i = prob_idx[j] - 1;  // Convert to 0-based
    if (obs_i >= 0 && obs_i < N) {
      prob_lookup[obs_i] = j;
    }
  }

  // Map to Cubes (no copy)
  cube eta_cube(eta_flat.begin(), n_topic - 1, n_cust, n_time, false);
  cube beta_cube(beta_flat.begin(), n_topic, n_var, n_item, false);
  cube x_cube(x_flat.begin(), n_var, n_time, n_item, false);

  // Step 1: Precompute Log-Likelihood tables [Z x T x I]
  cube log_phi_y1(n_topic, n_time, n_item);
  cube log_phi_y0(n_topic, n_time, n_item);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      vec xb_vec = beta_cube.slice(i) * x_cube.slice(i).col(t);
      for (int z = 0; z < n_topic; ++z) {
        double xb = xb_vec[z];
        log_phi_y1(z, t, i) = R::pnorm(xb, 0.0, 1.0, 1, 1);
        log_phi_y0(z, t, i) = R::pnorm(xb, 0.0, 1.0, 0, 1);
      }
    }
  }

  // Step 2: Main Sampling loop (sequential for prob_out write safety)
  vec log_probs(n_topic);
  vec probs(n_topic);

  for (int n = 0; n < N; ++n) {
    int c = cust_idx[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    int y = y_cit[n];

    double max_log_p = -1e308;

    for (int z = 0; z < n_topic; ++z) {
      double log_pi_z = (z < n_topic - 1) ? eta_cube(z, c, t) : 0.0;
      double log_omega = (y == 1) ? log_phi_y1(z, t, i) : log_phi_y0(z, t, i);
      log_probs[z] = log_pi_z + log_omega;
      if (log_probs[z] > max_log_p) max_log_p = log_probs[z];
    }

    // Convert to normalized probabilities
    double sum_p = 0.0;
    for (int z = 0; z < n_topic; ++z) {
      probs[z] = std::exp(log_probs[z] - max_log_p);
      sum_p += probs[z];
    }
    for (int z = 0; z < n_topic; ++z) {
      probs[z] /= sum_p;
    }

    // Store probabilities if this observation is tracked
    int prob_row = prob_lookup[n];
    if (prob_row >= 0) {
      for (int z = 0; z < n_topic; ++z) {
        prob_out(prob_row, z) = probs[z];
      }
    }

    // Categorical sampling via cumulative sum (use unnormalized weights)
    double r = rand_u[n];
    double cumulative_p = 0.0;
    int sampled_z = n_topic;
    for (int z = 0; z < n_topic; ++z) {
      cumulative_p += probs[z];
      if (r <= cumulative_p) {
        sampled_z = z + 1;
        break;
      }
    }
    z_new[n] = sampled_z;
  }

  return List::create(
    Named("z") = z_new,
    Named("prob") = prob_out
  );
}
