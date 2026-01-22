#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

/**
 * Helper function for Inverse-Wishart sampling.
 * Equivalent to MCMCpack::riwish.
 */
mat riwish_arma(double v, const mat& S) {
  mat S_inv = inv_sympd(S);
  mat L = chol(S_inv, "lower");
  // Draw from Wishart(v, S_inv) then invert
  mat X = randn<mat>(v, S.n_cols) * L.t();
  return inv_sympd(X.t() * X);
}

// [[Rcpp::export]]
List sample_mu_V_cpp(
    NumericVector beta_zi_flat,   // [n_topic, n_var, n_item]
    NumericMatrix mu_i_mat,       // [n_item, n_var]
    NumericVector V_i_flat,       // [n_var, n_var, n_item]
    arma::vec mu_tilde,           // Use arma:: prefix for RcppExport compatibility
    double sigma_tilde_mu,
    double w_tilde,
    arma::mat W_tilde,            // Use arma:: prefix for RcppExport compatibility
    int n_topic,
    int n_item,
    int n_var
) {
  // Map flattened vectors to cubes
  cube beta_zi_cube(beta_zi_flat.begin(), n_topic, n_var, n_item, false);
  cube V_i_cube(V_i_flat.begin(), n_var, n_var, n_item, true);
  mat mu_i_out(mu_i_mat.begin(), n_item, n_var, true);

#pragma omp parallel for
  for (int i = 0; i < n_item; ++i) {
    // 1. Calculate sample mean for item i
    mat beta_item_i = beta_zi_cube.slice(i);
    rowvec beta_bar = mean(beta_item_i, 0);

    // 2. Calculate Sum of Squared Deviations (S_data)
    mat S_data = zeros<mat>(n_var, n_var);
    for (int z = 0; z < n_topic; ++z) {
      rowvec diff = beta_item_i.row(z) - beta_bar;
      S_data += diff.t() * diff;
    }

    // --- 3. Update V_i (Marginal Posterior) ---
    double post_w = w_tilde + (double)n_topic;
    mat post_W = W_tilde + S_data;
    V_i_cube.slice(i) = riwish_arma(post_w, post_W);

    // --- 4. Update mu_i (Conditional Posterior) ---
    double prec_scale = (1.0 / sigma_tilde_mu) + (double)n_topic;
    mat Vi_curr = V_i_cube.slice(i);
    mat post_mu_cov = Vi_curr / prec_scale;

    vec post_mu_mean = (mu_tilde / sigma_tilde_mu + (double)n_topic * beta_bar.t()) / prec_scale;

    // Sample from Multivariate Normal
    vec eps = randn<vec>(n_var);
    mu_i_out.row(i) = (post_mu_mean + chol(post_mu_cov, "lower") * eps).t();
  }

  return List::create(
    Named("mu_i") = mu_i_out,
    Named("V_i")  = V_i_cube
  );
}
