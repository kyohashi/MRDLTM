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
#'
#' @return NULL
#' @importFrom pgdraw pgdraw
#' @importFrom stats rnorm
#' @noRd
sample_eta = function(active_data, state, Dc, n_cust, n_topic, length_time) {
  n_z_dlm = n_topic - 1

  # --- 1. Calculate n_kct and N_ct ---
  n_kct = array(0, dim = c(n_topic, n_cust, length_time))
  indices = cbind(state$z_cit, active_data$cust, active_data$time)
  for (i in 1:nrow(indices)) {
    n_kct[indices[i, 1], indices[i, 2], indices[i, 3]] =
      n_kct[indices[i, 1], indices[i, 2], indices[i, 3]] + 1
  }

  N_ct = apply(n_kct, c(2, 3), sum)
  # Find indices where purchases occurred to avoid pgdraw(b=0, ...)
  idx_pos = which(N_ct > 0)

  # Pre-calculate exp(eta) sum for all topics including the Z-th baseline (exp(0) = 1)
  exp_eta_all = exp(state$eta_zct) # Dimensions: [Z-1, C, T]
  sum_exp_total = apply(exp_eta_all, c(2, 3), sum) + 1

  # Loop through topics k = 1 to Z-1
  for (k in 1:n_z_dlm) {
    # --- 2. Update Polya-Gamma omega_kct ---
    # C_kct = 1 + sum_{j != k, j < Z} exp(eta_jct)
    log_C_kct = log(sum_exp_total - exp_eta_all[k, , ])

    # Vectorized draw with safety check for N_ct > 0
    omega_vec = rep(0, n_cust * length_time)
    if (length(idx_pos) > 0) {
      # Sample only where data exists
      psi_kct = (state$eta_zct[k, , ] - log_C_kct)[idx_pos]
      omega_vec[idx_pos] = pgdraw(N_ct[idx_pos], psi_kct)
    }
    state$omega_zct[k, , ] = matrix(omega_vec, nrow = n_cust, ncol = length_time)

    # --- 3. Update eta_kct ---
    # kappa_kct = n_kct - N_ct/2
    state$kappa_zct[k, , ] = n_kct[k, , ] - N_ct / 2

    # Precision and Mean for posterior Normal distribution
    a2_inv = 1 / state$a2_z[k]
    # Posterior precision: V_inv = omega + 1/a^2
    V_kct = 1 / (state$omega_zct[k, , ] + a2_inv)

    # Linear predictor from DLM: F_alpha = D_c * alpha_kt
    F_alpha = Dc %*% t(state$alpha_zt[k, , ]) # [C x T]

    # m_kct = V_kct * (omega * log_C + kappa + prior_mean / a^2)
    m_kct = V_kct * (state$omega_zct[k, , ] * log_C_kct + state$kappa_zct[k, , ] + F_alpha * a2_inv)

    # Sample eta_kct ~ N(m_kct, V_kct)
    state$eta_zct[k, , ] = matrix(
      rnorm(n_cust * length_time, mean = as.vector(m_kct), sd = sqrt(as.vector(V_kct))),
      nrow = n_cust, ncol = length_time
    )
  }
}
