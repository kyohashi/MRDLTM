#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List sample_eta_prepare_cpp(
    IntegerVector z_cit_flat,
    IntegerVector obs_cust,
    IntegerVector obs_time,
    NumericVector eta_zct_flat,
    int n_topic,
    int n_time,
    int n_cust
) {
  int n_z_dlm = n_topic - 1;
  cube eta_zct(eta_zct_flat.begin(), n_z_dlm, n_cust, n_time, false);

  // Count topic assignments: n_kct
  cube counts = zeros<cube>(n_topic, n_cust, n_time);
  for(int n = 0; n < z_cit_flat.size(); ++n) {
    counts(z_cit_flat[n]-1, obs_cust[n]-1, obs_time[n]-1) += 1.0;
  }

  int total_tasks = n_cust * n_time * n_z_dlm;
  NumericVector b_vec(total_tasks);
  NumericVector z_vec(total_tasks);
  LogicalVector needs_pg(total_tasks);

  // Prepare parameters for Polya-Gamma sampling
  for (int c = 0; c < n_cust; ++c) {
    for (int t = 0; t < n_time; ++t) {
      for (int k = 0; k < n_z_dlm; ++k) {
        int idx = c * (n_time * n_z_dlm) + t * n_z_dlm + k;

        // m_kt = sum_{j=k}^Z n_jt (Total counts for topics >= k)
        double m_kt = 0;
        for(int j = k; j < n_topic; ++j) m_kt += counts(j, c, t);

        b_vec[idx] = m_kt;
        z_vec[idx] = eta_zct(k, c, t);
        needs_pg[idx] = (m_kt > 0);
      }
    }
  }

  return List::create(
    Named("b_vec") = b_vec,
    Named("z_vec") = z_vec,
    Named("needs_pg") = needs_pg,
    Named("counts") = counts
  );
}

// [[Rcpp::export]]
NumericVector sample_eta_update_cpp(
    NumericVector eta_zct_flat,
    NumericVector omega_vec,
    NumericVector counts_flat,
    NumericVector alpha_zt_flat,
    NumericMatrix Dc_mat,
    arma::vec b2_z,
    int n_topic,
    int n_time,
    int n_cust,
    int p_dim
) {
  int n_z_dlm = n_topic - 1;
  cube eta_zct(eta_zct_flat.begin(), n_z_dlm, n_cust, n_time, true);
  cube counts(counts_flat.begin(), n_topic, n_cust, n_time, false);
  cube alpha_zt(alpha_zt_flat.begin(), n_z_dlm, n_time, p_dim, false);
  mat Dc(Dc_mat.begin(), n_cust, p_dim, false);

  // Parallelize posterior updates across customers and time points
#pragma omp parallel for collapse(2)
  for (int c = 0; c < n_cust; ++c) {
    for (int t = 0; t < n_time; ++t) {
      rowvec d_c = Dc.row(c);
      for (int k = 0; k < n_z_dlm; ++k) {
        int idx = c * (n_time * n_z_dlm) + t * n_z_dlm + k;
        double omega = omega_vec[idx];
        double current_b2 = b2_z[k];

        double n_kt = counts(k, c, t);
        double m_kt = 0;
        for(int j = k; j < n_topic; ++j) m_kt += counts(j, c, t);

        // mu_kct = d_c * alpha_kt (DLM prior mean)
        double mu_kct = as_scalar(d_c * vectorise(alpha_zt.tube(k, t)));

        if (m_kt > 0) {
          double kappa = n_kt - 0.5 * m_kt;
          double post_var = 1.0 / (omega + 1.0 / current_b2);
          double post_mean = post_var * (kappa + mu_kct / current_b2);

          eta_zct(k, c, t) = post_mean + std::sqrt(post_var) * R::rnorm(0, 1);
        } else {
          // No observations, sample from the DLM prior
          eta_zct(k, c, t) = mu_kct + std::sqrt(current_b2) * R::rnorm(0, 1);
        }
      }
    }
  }
  return wrap(eta_zct);
}
