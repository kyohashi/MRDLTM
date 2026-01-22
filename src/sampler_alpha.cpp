#include <RcppArmadillo.h>
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
    int n_topic,
    int n_time,
    int n_cust,
    int p_dim
) {
  int n_z_dlm = n_topic - 1;
  mat Dc(Dc_mat.begin(), n_cust, p_dim, false);
  mat DtD = Dc.t() * Dc;
  vec mz0(mz0_vec.begin(), p_dim, false);
  mat Sz0(Sz0_mat.begin(), p_dim, p_dim, false);

  // Result storage: [n_z_dlm, n_time, p_dim]
  cube alpha_new(n_z_dlm, n_time, p_dim);

  // Convert flattened eta to a more accessible structure
  // Pre-calculating all D' * eta_t for all topics and times
  field<mat> Dt_eta_all(n_z_dlm);
  for(int k = 0; k < n_z_dlm; ++k) {
    mat eta_k(n_cust, n_time);
    for(int t = 0; t < n_time; ++t) {
      for(int c = 0; c < n_cust; ++c) {
        eta_k(c, t) = eta_zct_flat[k + c * n_z_dlm + t * n_z_dlm * n_cust];
      }
    }
    Dt_eta_all(k) = Dc.t() * eta_k; // Result is [p_dim, n_time]
  }

  // Parallelize across topics
#pragma omp parallel for
  for (int k = 0; k < n_z_dlm; ++k) {
    double a2 = a2_z[k];
    double b2 = b2_z[k];
    mat W = eye<mat>(p_dim, p_dim) * b2;
    mat DtD_a2 = DtD / a2;

    // Local buffers for Filtering
    std::vector<vec> m_filt(n_time + 1);
    std::vector<mat> C_filt(n_time + 1);

    m_filt[0] = mz0;
    C_filt[0] = Sz0;

    // --- Forward Filtering ---
    for (int t = 1; t <= n_time; ++t) {
      vec m_pred = m_filt[t-1];
      mat R_pred = C_filt[t-1] + W;
      mat R_inv = inv_sympd(R_pred);

      vec info_mean = Dt_eta_all(k).col(t-1) / a2;

      mat C_curr_inv = R_inv + DtD_a2;
      mat C_curr = inv_sympd(C_curr_inv);
      vec m_curr = C_curr * (R_inv * m_pred + info_mean);

      m_filt[t] = m_curr;
      C_filt[t] = C_curr;
    }

    // --- Backward Sampling ---
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
