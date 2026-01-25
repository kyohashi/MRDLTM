#include <RcppArmadillo.h>
#include <vector>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List sample_dlm_vars_cpp(
    NumericVector eta_zct_flat,
    NumericVector alpha_zt_flat,
    NumericMatrix Dc_mat,
    IntegerVector obs_cust,
    IntegerVector obs_time,
    double a2_prior_shape,
    double a2_prior_scale,
    double b2_prior_shape,
    double b2_prior_scale,
    int n_topic,
    int n_time,
    int n_cust,
    int p_dim
) {
  int n_z_dlm = n_topic - 1;
  // Map flattened vectors to cubes/matrices
  // eta_zct: [Topic, Cust, Time]
  cube eta_zct(eta_zct_flat.begin(), n_z_dlm, n_cust, n_time, false);
  cube alpha_zt(alpha_zt_flat.begin(), n_z_dlm, n_time, p_dim, false);
  mat Dc(Dc_mat.begin(), n_cust, p_dim, false);

  vec a2_new(n_z_dlm);
  vec b2_new(n_z_dlm);

  // --- 1. Pre-process Active Indices per Time Point ---
  // Identify which customers are active at each time point t.
  std::vector<std::set<int>> active_sets(n_time);
  int n_obs = obs_cust.size();

  for(int n = 0; n < n_obs; ++n) {
    int t = obs_time[n] - 1;
    int c = obs_cust[n] - 1;
    if (t >= 0 && t < n_time && c >= 0 && c < n_cust) {
      active_sets[t].insert(c);
    }
  }

  // Convert to Armadillo uvec for slicing
  std::vector<uvec> active_indices(n_time);
  for(int t = 0; t < n_time; ++t) {
    if(!active_sets[t].empty()) {
      active_indices[t] = uvec(active_sets[t].size());
      int idx = 0;
      for(int c : active_sets[t]) {
        active_indices[t](idx++) = (uword)c;
      }
    }
  }

#pragma omp parallel for
  for (int k = 0; k < n_z_dlm; ++k) {

    // --- Update a2_z (Observation Variance) ---
    // We construct eta_k [n_cust x n_time] for easy column slicing
    mat eta_k(n_cust, n_time);
    for(int t = 0; t < n_time; ++t) {
      for(int c = 0; c < n_cust; ++c) {
        eta_k(c, t) = eta_zct(k, c, t);
      }
    }

    double ssr_a = 0.0;
    int active_count = 0;

    for (int t = 0; t < n_time; ++t) {
      uvec current_indices = active_indices[t];

      if (current_indices.n_elem > 0) {
        // 1. Subset Dc for active customers
        mat Dc_sub = Dc.rows(current_indices);

        // 2. Get alpha for this topic and time
        vec alpha_kt = vectorise(alpha_zt.tube(k, t));

        // 3. Compute Prediction: Dc_sub * alpha_t
        vec pred_eta = Dc_sub * alpha_kt;

        // 4. Subset observed Eta
        // Use submat for safe column slicing with uvec
        uvec t_idx = { (uword)t };
        vec obs_eta = eta_k.submat(current_indices, t_idx);

        // 5. Compute Residuals
        vec resid = obs_eta - pred_eta;
        ssr_a += dot(resid, resid);

        // 6. Accumulate sample size
        active_count += current_indices.n_elem;
      }
    }

    // Update Step for a2
    // Shape uses actual number of observations (active_count), NOT (n_cust * n_time)
    double shape_a = a2_prior_shape + 0.5 * (double)active_count;
    double scale_a = a2_prior_scale + 0.5 * ssr_a;

    // Safety for zero observations (prior only)
    if (active_count == 0) {
      shape_a = a2_prior_shape;
      scale_a = a2_prior_scale;
    }

    a2_new[k] = 1.0 / R::rgamma(shape_a, 1.0 / scale_a);


    // --- Update b2_z (System Variance) ---
    // This depends only on alpha transitions, not on observations.
    // So we use the full time series of alpha.
    double ssr_b = 0.0;
    for (int t = 1; t < n_time; ++t) {
      vec alpha_t = vectorise(alpha_zt.tube(k, t));
      vec alpha_prev = vectorise(alpha_zt.tube(k, t - 1));
      vec diff = alpha_t - alpha_prev;
      ssr_b += dot(diff, diff);
    }

    // Prior assumes system noise is b2 * I_p
    double shape_b = b2_prior_shape + 0.5 * (double)(p_dim * (n_time - 1));
    double scale_b = b2_prior_scale + 0.5 * ssr_b;
    b2_new[k] = 1.0 / R::rgamma(shape_b, 1.0 / scale_b);
  }

  return List::create(
    Named("a2_z") = a2_new,
    Named("b2_z") = b2_new
  );
}
