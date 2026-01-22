#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double compute_log_likelihood_cpp(
    IntegerVector z_cit,      // Topic assignments [n_obs]
    IntegerVector item_idx,   // Item indices [n_obs]
    IntegerVector time_idx,   // Time indices [n_obs]
    IntegerVector y_cit,      // Binary observations [n_obs]
    NumericVector beta_zi_flat, // [n_topic, n_item, n_var]
    NumericVector x_it_flat,    // [n_item, n_time, n_var]
    int n_topic,
    int n_item,
    int n_time,
    int n_var
) {
  // Map flattened arrays to cubes for efficient indexing
  cube beta_zi(beta_zi_flat.begin(), n_topic, n_item, n_var, false);
  cube x_it(x_it_flat.begin(), n_item, n_time, n_var, false);

  int n_obs = z_cit.size();
  double total_log_lik = 0.0;

  // Parallelize the observation loop
#pragma omp parallel for reduction(+:total_log_lik)
  for (int n = 0; n < n_obs; ++n) {
    // Convert 1-based R indices to 0-based C++ indices
    int z = z_cit[n] - 1;
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    int y = y_cit[n];

    // Linear predictor: dot product of x_it[i, t, ] and beta_zi[z, i, ]
    double xb = 0.0;
    for (int m = 0; m < n_var; ++m) {
      xb += x_it(i, t, m) * beta_zi(z, i, m);
    }

    // Log-Likelihood: Log Phi(xb) if y=1, Log(1-Phi(xb)) if y=0
    // R::pnorm(x, mu, sigma, lower_tail, log_p)
    total_log_lik += R::pnorm(xb, 0.0, 1.0, (y == 1), 1);
  }

  return total_log_lik;
}
