#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector sample_u_cpp(
    IntegerVector y_cit,
    NumericMatrix x_it_matrix,
    NumericVector beta_zi_flat,
    IntegerVector z_cit,
    IntegerVector item_idx,
    IntegerVector time_idx,
    int n_topic,
    int n_item,
    int n_var
) {
  int N = y_cit.size();
  NumericVector u_new(N);

  for (int n = 0; n < N; ++n) {
    int i = item_idx[n] - 1;
    int t = time_idx[n] - 1;
    int z = z_cit[n] - 1;
    int y = y_cit[n];

    double xb = 0.0;
    for (int v = 0; v < n_var; ++v) {
      xb += x_it_matrix(i + t * n_item, v) * beta_zi_flat[z + i * n_topic + v * n_topic * n_item];
    }

    double p_lower, p_upper;
    if (y == 1) {
      p_lower = R::pnorm(0.0, xb, 1.0, 1, 0);
      p_upper = 1.0;
    } else {
      p_lower = 0.0;
      p_upper = R::pnorm(0.0, xb, 1.0, 1, 0);
    }

    double u_rand = R::runif(0, 1);
    double target_p = p_lower + u_rand * (p_upper - p_lower);

    if (target_p < 0.0000001) target_p = 0.0000001;
    if (target_p > 0.9999999) target_p = 0.9999999;

    u_new[n] = R::qnorm(target_p, xb, 1.0, 1, 0);
  }

  return u_new;
}
