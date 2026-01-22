#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
NumericVector sample_beta_cpp(
    NumericVector u_cit,
    NumericMatrix x_it_matrix,
    IntegerVector z_cit,
    IntegerVector item_idx,
    IntegerVector time_idx,
    NumericMatrix mu_beta_zi,
    NumericVector V_inv_zi, // Match the R argument name exactly
    int n_topic,
    int n_item,
    int n_var
) {
  field<mat> XX(n_topic, n_item);
  field<vec> Xy(n_topic, n_item);

  for(int z = 0; z < n_topic; z++) {
    for(int i = 0; i < n_item; i++) {
      XX(z, i) = zeros<mat>(n_var, n_var);
      Xy(z, i) = zeros<vec>(n_var);
    }
  }

  int N = u_cit.size();
  mat X_mat(x_it_matrix.begin(), x_it_matrix.nrow(), x_it_matrix.ncol(), false);

  for (int n = 0; n < N; ++n) {
    int z = z_cit[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;

    rowvec x = X_mat.row(i + t * n_item);
    double u = u_cit[n];

    XX(z, i) += x.t() * x;
    Xy(z, i) += x.t() * u;
  }

  // V_inv_zi is flattened [n_var, n_var, n_item]
  cube V_inv_cube(V_inv_zi.begin(), n_var, n_var, n_item, false);
  mat mu_prior_mat(mu_beta_zi.begin(), mu_beta_zi.nrow(), mu_beta_zi.ncol(), false);

  cube beta_new(n_topic, n_item, n_var);

  for (int i = 0; i < n_item; ++i) {
    mat V_inv = V_inv_cube.slice(i);
    vec mu_prior = mu_prior_mat.row(i).t();

    for (int z = 0; z < n_topic; ++z) {
      mat V_post_inv = XX(z, i) + V_inv;
      mat V_post = inv_sympd(V_post_inv);
      vec mu_post = V_post * (Xy(z, i) + V_inv * mu_prior);

      vec eps = randn<vec>(n_var);
      beta_new.tube(z, i) = mu_post + chol(V_post, "lower") * eps;
    }
  }

  return wrap(beta_new);
}
