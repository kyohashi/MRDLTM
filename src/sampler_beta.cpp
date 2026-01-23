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
  cube x_it(x_it_flat.begin(), n_item, n_time, n_var, false);
  mat mu_i(mu_i_mat.begin(), n_item, n_var, false);
  cube V_i(V_i_flat.begin(), n_var, n_var, n_item, false);

  // 1. Aggregation Step (O(N)): Collect scalars first (Fast)
  // Use a flat vector to store sums of u and counts to avoid field<> overhead
  arma::cube sum_u(n_topic, n_item, n_time, fill::zeros);
  arma::cube counts(n_topic, n_item, n_time, fill::zeros);

  for (int n = 0; n < n_obs; ++n) {
    sum_u(z_cit[n]-1, item_idx[n]-1, time_idx[n]-1) += u_cit[n];
    counts(z_cit[n]-1, item_idx[n]-1, time_idx[n]-1) += 1.0;
  }

  // 2. Statistics Step (O(K*I*T)): Matrix operations on aggregated data
  field<mat> XX(n_topic, n_item);
  field<vec> Xu(n_topic, n_item);

  // Initialize with Priors
  for(int k = 0; k < n_topic; ++k) {
    for(int i = 0; i < n_item; ++i) {
      mat V_inv = inv_sympd(V_i.slice(i));
      XX(k, i) = V_inv;
      Xu(k, i) = V_inv * mu_i.row(i).t();
    }
  }

  // Heavy matrix math happens here, but only K*I*T times (much less than N)
  for (int i = 0; i < n_item; ++i) {
    for (int t = 0; t < n_time; ++t) {
      vec x_n = vectorise(x_it.tube(i, t));
      mat x_outer = x_n * x_n.t(); // Compute outer product only once per (i, t)

      for (int k = 0; k < n_topic; ++k) {
        if (counts(k, i, t) > 0) {
          XX(k, i) += counts(k, i, t) * x_outer;
          Xu(k, i) += x_n * sum_u(k, i, t);
        }
      }
    }
  }

  // 3. Sampling Step: Parallel as before
  cube beta_out(n_topic, n_item, n_var);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n_item; ++i) {
    for (int k = 0; k < n_topic; ++k) {
      mat Post_Cov = inv_sympd(XX(k, i));
      beta_out.tube(k, i) = Post_Cov * Xu(k, i) + chol(Post_Cov, "lower") * randn<vec>(n_var);
    }
  }
  return wrap(beta_out);
}
