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
    int n_topic, int n_time, int n_cust
) {
  int n_z_dlm = n_topic - 1;
  cube eta_zct(eta_zct_flat.begin(), n_z_dlm, n_cust, n_time, false);

  // Aggregate counts n_kct
  cube counts = zeros<cube>(n_topic, n_cust, n_time);
  for(int n = 0; n < z_cit_flat.size(); ++n) {
    counts(z_cit_flat[n]-1, obs_cust[n]-1, obs_time[n]-1) += 1.0;
  }

  int total_tasks = n_cust * n_time * n_z_dlm;
  NumericVector b_vec(total_tasks);
  NumericVector z_vec(total_tasks);
  NumericVector log_C_vec(total_tasks);
  NumericVector n_kct_vec(total_tasks);
  NumericVector N_ct_vec(total_tasks);
  LogicalVector needs_pg(total_tasks); // Fixed: Re-added logical vector

  for (int c = 0; c < n_cust; ++c) {
    for (int t = 0; t < n_time; ++t) {
      // N_ct: Total samples for customer c at time t
      double N_ct = 0;
      for(int j = 0; j < n_topic; ++j) N_ct += counts(j, c, t);

      // exp_sum = 1 + sum_{j=1}^{Z-1} exp(eta_jct)
      double exp_sum = 1.0;
      for (int j = 0; j < n_z_dlm; ++j) exp_sum += std::exp(eta_zct(j, c, t));

      for (int k = 0; k < n_z_dlm; ++k) {
        int idx = c * (n_time * n_z_dlm) + t * n_z_dlm + k;

        // C_kct = exp_sum - exp(eta_kct) (as per Eq 49)
        double C_kct = exp_sum - std::exp(eta_zct(k, c, t));
        double log_Ckct = std::log(C_kct);

        b_vec[idx] = N_ct;
        z_vec[idx] = eta_zct(k, c, t) - log_Ckct; // psi_kct
        log_C_vec[idx] = log_Ckct;
        n_kct_vec[idx] = counts(k, c, t);
        N_ct_vec[idx] = N_ct;
        needs_pg[idx] = (N_ct > 0);
      }
    }
  }

  return List::create(
    Named("b_vec") = b_vec,
    Named("z_vec") = z_vec,
    Named("log_C_vec") = log_C_vec,
    Named("n_kct_vec") = n_kct_vec,
    Named("N_ct_vec") = N_ct_vec,
    Named("needs_pg") = needs_pg, // Fixed: Added to list
    Named("counts") = counts
  );
}

// [[Rcpp::export]]
NumericVector sample_eta_update_cpp(
    NumericVector eta_zct_flat,
    NumericVector omega_vec,
    NumericVector log_C_vec,
    NumericVector n_kct_vec,
    NumericVector N_ct_vec,
    NumericVector alpha_zt_flat,
    NumericMatrix Dc_mat,
    arma::vec a2_z,
    int n_topic, int n_time, int n_cust, int p_dim
) {
  int n_z_dlm = n_topic - 1;
  cube eta_zct(eta_zct_flat.begin(), n_z_dlm, n_cust, n_time, true);
  cube alpha_zt(alpha_zt_flat.begin(), n_z_dlm, n_time, p_dim, false);
  mat Dc(Dc_mat.begin(), n_cust, p_dim, false);

#pragma omp parallel for collapse(2)
  for (int c = 0; c < n_cust; ++c) {
    for (int t = 0; t < n_time; ++t) {
      rowvec d_c = Dc.row(c);
      for (int k = 0; k < n_z_dlm; ++k) {
        int idx = c * (n_time * n_z_dlm) + t * n_z_dlm + k;

        double omega = omega_vec[idx];
        double log_Ckct = log_C_vec[idx];
        double n_kct = n_kct_vec[idx];
        double N_ct = N_ct_vec[idx];
        double var_obs = a2_z[k]; // observation variance a_k^2

        // kappa_kct = n_kct - N_ct / 2
        double kappa = n_kct - 0.5 * N_ct;
        double mu_prior = as_scalar(d_c * vectorise(alpha_zt.tube(k, t)));

        // Equation (66) and (67)
        double V_post = 1.0 / (omega + 1.0 / var_obs);
        double m_post = V_post * (omega * log_Ckct + kappa + mu_prior / var_obs);

        eta_zct(k, c, t) = m_post + std::sqrt(V_post) * R::rnorm(0, 1);
      }
    }
  }
  return wrap(eta_zct);
}
