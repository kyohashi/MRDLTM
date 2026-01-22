#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
NumericVector sample_beta_cpp(
    IntegerVector z_cit,      // Topic assignments [n_obs]
    IntegerVector item_idx,   // Item indices [n_obs]
    IntegerVector time_idx,   // Time indices [n_obs]
    NumericVector u_cit,      // Latent utilities [n_obs]
    NumericVector x_it_flat,  // Covariates [n_item, n_time, n_var]
    NumericMatrix mu_i_mat,   // Hierarchical mean [n_item, n_var]
    NumericVector V_i_flat,   // Hierarchical variance [n_var, n_var, n_item]
    int n_topic,
    int n_item,
    int n_time,
    int n_var
) {
  int n_obs = z_cit.size();

  // x_it is [I, T, M] in R, so map to cube(I, T, M)
  cube x_it(x_it_flat.begin(), n_item, n_time, n_var, false);
  mat mu_i(mu_i_mat.begin(), n_item, n_var, false);
  // V_i is [M, M, I] after aperm in R
  cube V_i(V_i_flat.begin(), n_var, n_var, n_item, false);

  // Storage for sufficient statistics XX (M x M) and Xu (M x 1)
  field<mat> XX(n_topic, n_item);
  field<vec> Xu(n_topic, n_item);

  // Initialize using hierarchical priors
  for(int k = 0; k < n_topic; ++k) {
    for(int i = 0; i < n_item; ++i) {
      mat V_inv = inv_sympd(V_i.slice(i));
      XX(k, i) = V_inv;
      Xu(k, i) = V_inv * mu_i.row(i).t();
    }
  }

  // O(N) Step: Single pass over observations
  for (int n = 0; n < n_obs; ++n) {
    int k = z_cit[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;

    // Extract covariate vector of length M for specific (i, t)
    // tube(i, t) returns all elements along the 3rd dimension
    vec x_n = vectorise(x_it.tube(i, t));

    XX(k, i) += x_n * x_n.t();
    Xu(k, i) += x_n * u_cit[n];
  }

  // Result cube: [n_topic, n_item, n_var] to match R's array(Z, I, M)
  cube beta_out(n_topic, n_item, n_var);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int k = 0; k < n_topic; ++k) {
      mat Post_Cov = inv_sympd(XX(k, i));
      vec Post_Mean = Post_Cov * Xu(k, i);

      vec eps = randn<vec>(n_var);
      // Store as tube (length M) at (k, i) to preserve memory layout
      beta_out.tube(k, i) = Post_Mean + chol(Post_Cov, "lower") * eps;
    }
  }

  return wrap(beta_out);
}
