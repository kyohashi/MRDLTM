#' Sample Latent Occupancy eta_zct
#'
#' @description
#' Updates the auxiliary Polya-Gamma variables (omega) and the latent occupancy
#' parameters (eta) using the PG-augmentation method.
#'
#' @param active_data A data frame of active observations (cust, item, time, y_cit).
#' @param state An environment containing the current MCMC state.
#' @param Dc Matrix of customer covariates (n_cust x p_dim).
#' @param n_cust Total number of customers (C).
#' @param n_topic Total number of topics (Z).
#' @param length_time Total length of time points (T).
#' @param p_dim Dimension of customer covariates (P).
#'
#' @return NULL
#' @importFrom pgdraw pgdraw
#' @noRd
sample_eta = function(active_data, state, Dc, n_cust, n_topic, length_time, p_dim) {

  # Step 1: Prepare parameters for PG sampling in C++ (fast)
  prep = sample_eta_prepare_cpp(
    z_cit_flat   = as.integer(state$z_cit),
    obs_cust     = as.integer(active_data$cust),
    obs_time     = as.integer(active_data$time),
    eta_zct_flat = as.numeric(state$eta_zct),
    n_topic      = n_topic,
    n_time       = length_time,
    n_cust       = n_cust
  )

  # Step 2: Sample Polya-Gamma variables as a single vector
  omega_vec = rep(0, length(prep$b_vec))
  active_idx = which(prep$needs_pg)
  if(length(active_idx) > 0) {
    # Call vectorized pgdraw to avoid R-C++ overhead inside loops
    omega_vec[active_idx] = pgdraw(prep$b_vec[active_idx], prep$z_vec[active_idx])
  }

  # Step 3: Update posterior mean/variance in C++ (parallelized)
  state$eta_zct = sample_eta_update_cpp(
    eta_zct_flat  = as.numeric(state$eta_zct),
    omega_vec     = omega_vec,
    counts_flat   = as.numeric(prep$counts),
    alpha_zt_flat = as.numeric(state$alpha_zt),
    Dc_mat        = as.matrix(Dc),
    b2_z          = as.numeric(state$b2_z),
    n_topic       = n_topic,
    n_time        = length_time,
    n_cust        = n_cust,
    p_dim         = p_dim
  )
}
