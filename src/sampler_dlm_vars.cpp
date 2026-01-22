#include <RcppArmadillo.h>
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
  cube eta_zct(eta_zct_flat.begin(), n_z_dlm, n_cust, n_time, false);
  cube alpha_zt(alpha_zt_flat.begin(), n_z_dlm, n_time, p_dim, false);
  mat Dc(Dc_mat.begin(), n_cust, p_dim, false);

  vec a2_new(n_z_dlm);
  vec b2_new(n_z_dlm);

  // Pre-calculate D * alpha for all topics and time points
  // Using a field of matrices to store [n_cust x 1] results
  field<mat> D_alpha(n_z_dlm, n_time);
  for(int k = 0; k < n_z_dlm; ++k) {
    for(int t = 0; t < n_time; ++t) {
      // Explicitly convert subview_cube to vec to avoid operator* ambiguity
      vec alpha_kt = vectorise(alpha_zt.tube(k, t));
      D_alpha(k, t) = Dc * alpha_kt;
    }
  }

#pragma omp parallel for
  for (int k = 0; k < n_z_dlm; ++k) {
    // --- 1. Update a2_z (Observation Variance) ---
    double ssr_a = 0.0;
    for (int t = 0; t < n_time; ++t) {
      for (int c = 0; c < n_cust; ++c) {
        double resid = eta_zct(k, c, t) - D_alpha(k, t)(c);
        ssr_a += resid * resid;
      }
    }
    double shape_a = a2_prior_shape + 0.5 * (n_cust * n_time);
    double scale_a = a2_prior_scale + 0.5 * ssr_a;
    // Sample from Inverse Gamma: 1 / Gamma(shape, 1/scale)
    a2_new[k] = 1.0 / R::rgamma(shape_a, 1.0 / scale_a);

    // --- 2. Update b2_z (System Variance) ---
    double ssr_b = 0.0;
    for (int t = 1; t < n_time; ++t) {
      // Explicitly convert to vec for safe subtraction and dot product
      vec alpha_t = vectorise(alpha_zt.tube(k, t));
      vec alpha_prev = vectorise(alpha_zt.tube(k, t - 1));
      vec diff = alpha_t - alpha_prev;
      ssr_b += dot(diff, diff);
    }

    // Prior assumes system noise is b2 * I_p
    double shape_b = b2_prior_shape + 0.5 * (p_dim * (n_time - 1));
    double scale_b = b2_prior_scale + 0.5 * ssr_b;
    b2_new[k] = 1.0 / R::rgamma(shape_b, 1.0 / scale_b);
  }

  return List::create(
    Named("a2_z") = a2_new,
    Named("b2_z") = b2_new
  );
}
