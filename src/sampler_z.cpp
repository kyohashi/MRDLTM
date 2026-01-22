#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector sample_z_cpp(NumericVector u_cit,
                           NumericVector prob_matrix_flat,
                           int n_topic) {
  // ここに爆速のループ処理を書いていきます
  // まずはコンパイルが通るか確認するための骨組みです
  int N = u_cit.size();
  IntegerVector z_new(N);
  return z_new;
}
