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
NumericVector sample_alpha_cpp(
    NumericVector eta_zct_flat,
    NumericMatrix Dc_mat,
    NumericVector a2_z,
    NumericVector b2_z,
    NumericVector mz0_vec,
    NumericMatrix Sz0_mat,
    IntegerVector obs_cust,
    IntegerVector obs_time,
    int n_topic,
    int n_time,
    int n_cust,
    int p_dim
) {
  int n_z_dlm = n_topic - 1;
  mat Dc(Dc_mat.begin(), n_cust, p_dim, false);
  vec mz0(mz0_vec.begin(), p_dim, false);
  mat Sz0(Sz0_mat.begin(), p_dim, p_dim, false);

  // --- 1. Pre-process Active Indices per Time Point ---
  // We need to identify which customers are active at each time t.
  // This allows us to filter the observation equation dynamically.
  std::vector<std::set<int>> active_sets(n_time);
  int n_obs = obs_cust.size();

  for(int n = 0; n < n_obs; ++n) {
    // Convert 1-based R indices to 0-based C++ indices
    int t = obs_time[n] - 1;
    int c = obs_cust[n] - 1;
    if (t >= 0 && t < n_time && c >= 0 && c < n_cust) {
      active_sets[t].insert(c);
    }
  }

  // Convert sets to Armadillo uvec for efficient slicing
  std::vector<uvec> active_indices(n_time);
  for(int t = 0; t < n_time; ++t) {
    if(!active_sets[t].empty()) {
      active_indices[t] = uvec(active_sets[t].size());
      int idx = 0;
      for(int c : active_sets[t]) {
        active_indices[t](idx++) = (uword)c;
      }
    }
    // If empty, active_indices[t] remains 0-sized uvec
  }

  // Result storage: [n_z_dlm, n_time, p_dim]
  cube alpha_new(n_z_dlm, n_time, p_dim);

  // Parallelize across topics
#pragma omp parallel for
  for (int k = 0; k < n_z_dlm; ++k) {

    // Extract full eta for this topic first (to allow slicing inside the loop)
    // Layout of eta_zct_flat is assumed: [Topic, Cust, Time] (column-major in R)
    // We reconstruct the [Cust x Time] matrix for the current topic k.
    mat eta_k(n_cust, n_time);
    for(int t = 0; t < n_time; ++t) {
      for(int c = 0; c < n_cust; ++c) {
        eta_k(c, t) = eta_zct_flat[k + c * n_z_dlm + t * n_z_dlm * n_cust];
      }
    }

    double a2 = a2_z[k];
    double b2 = b2_z[k];
    mat W = eye<mat>(p_dim, p_dim) * b2;

    // Local buffers for Filtering
    std::vector<vec> m_filt(n_time + 1);
    std::vector<mat> C_filt(n_time + 1);

    // Initial State
    m_filt[0] = mz0;
    C_filt[0] = Sz0;

    // --- Forward Filtering ---
    for (int t = 1; t <= n_time; ++t) {
      // 1. Prediction Step (Time Update)
      vec m_pred = m_filt[t-1];        // G_t = I
      mat R_pred = C_filt[t-1] + W;    // G_t = I

      // 2. Measurement Update (only if data exists)
      uvec current_indices = active_indices[t-1];

      if (current_indices.n_elem > 0) {
        // Subset data for active customers
        mat Dc_sub = Dc.rows(current_indices);
        vec eta_sub = eta_k.submat(current_indices, uvec{(uword)(t-1)});

        // Compute observation precision components
        mat DtD_sub = Dc_sub.t() * Dc_sub;
        vec info_mean = Dc_sub.t() * eta_sub / a2;

        mat R_inv = inv_sympd(R_pred);
        mat C_curr_inv = R_inv + DtD_sub / a2;

        mat C_curr = inv_sympd(C_curr_inv);
        vec m_curr = C_curr * (R_inv * m_pred + info_mean);

        m_filt[t] = m_curr;
        C_filt[t] = C_curr;

      } else {
        // No data at this time point: Posterior = Prior (Prediction)
        m_filt[t] = m_pred;
        C_filt[t] = R_pred;
      }
    }

    // --- Backward Sampling (FFBS) ---
    // Seed RNG locally if needed, but Rcpp::RNGScope covers this
    vec alpha_next = m_filt[n_time] + chol(C_filt[n_time], "lower") * randn<vec>(p_dim);
    alpha_new.tube(k, n_time - 1) = alpha_next;

    for (int t = n_time - 1; t >= 1; --t) {
      mat R_next_pred = C_filt[t] + W;
      mat R_inv = inv_sympd(R_next_pred);
      mat gain = C_filt[t] * R_inv;

      vec m_back = m_filt[t] + gain * (alpha_next - m_filt[t]);
      mat V_back = C_filt[t] - gain * C_filt[t];
      V_back = (V_back + V_back.t()) * 0.5; // Ensure symmetry

      alpha_next = m_back + chol(V_back, "lower") * randn<vec>(p_dim);
      alpha_new.tube(k, t - 1) = alpha_next;
    }
  }

  return wrap(alpha_new);
}
