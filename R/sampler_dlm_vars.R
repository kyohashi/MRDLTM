#' Sample DLM Variance Parameters (a2_z and b2_z)
#'
#' @description
#' Samples the observation variance (a2_z) and system variance (b2_z)
#' for the dynamic topic model.
#'
#' @param state An environment containing the current MCMC state.
#' @param Dc Matrix of customer covariates (n_cust x p_dim).
#' @param n_topic Total number of topics (Z).
#' @param length_time Total length of time points (T).
#' @param p_dim Dimension of customer covariates (P).
#' @param priors A list of hyperparameters from the model object.
#'
#' @return NULL
#' @importFrom stats rgamma
#' @noRd
sample_dlm_vars = function(state, Dc, n_topic, length_time, p_dim, priors) {
  n_z_dlm = n_topic - 1
  n_cust = nrow(Dc)

  # --- 1. Hyperparameters ---
  # Observation variance priors (a2_z)
  nu_a0    = if (!is.null(priors$nu_a0)) priors$nu_a0 else 0.01
  delta_a0 = if (!is.null(priors$delta_a0)) priors$delta_a0 else 0.01

  # System variance priors (b2_z)
  nu_b0    = if (!is.null(priors$nu_b0)) priors$nu_b0 else 0.01
  delta_b0 = if (!is.null(priors$delta_b0)) priors$delta_b0 else 0.01

  # Initial state mean (mz0) for system residuals at t=1
  mz0 = if (!is.null(priors$mz0)) priors$mz0 else rep(0, p_dim)

  for (k in 1:n_z_dlm) {
    # --- 2. Sample a2_z (Observation Variance) ---
    # Observation residuals: eta_kct - D_c %*% alpha_kt
    eta_pred = Dc %*% t(state$alpha_zt[k, , ]) # [C x T]
    residuals_a = (state$eta_zct[k, , ] - eta_pred)^2

    # nu_an = nu_a0 + C * T
    nu_an = nu_a0 + (n_cust * length_time)
    # delta_an = delta_a0 + sum(residuals^2)
    delta_an = delta_a0 + sum(residuals_a)

    # Inverse-Gamma sampling
    state$a2_z[k] = 1 / rgamma(1, shape = nu_an / 2, rate = delta_an / 2)

    # --- 3. Sample b2_z (System/Evolution Variance) ---
    # System residuals: alpha_kt - alpha_{k,t-1}
    # For t=1, we use (alpha_k1 - mz0)
    alpha_k_mat = matrix(state$alpha_zt[k, , ], nrow = length_time, ncol = p_dim)

    diff_alpha = matrix(0, nrow = length_time, ncol = p_dim)
    diff_alpha[1, ] = alpha_k_mat[1, ] - mz0
    if (length_time > 1) {
      diff_alpha[2:length_time, ] = alpha_k_mat[2:length_time, ] - alpha_k_mat[1:(length_time-1), ]
    }

    # nu_bn = nu_b0 + T * P
    nu_bn = nu_b0 + (length_time * p_dim)
    # delta_bn = delta_b0 + sum(diff^2)
    delta_bn = delta_b0 + sum(diff_alpha^2)

    # Inverse-Gamma sampling
    state$b2_z[k] = 1 / stats::rgamma(1, shape = nu_bn / 2, rate = delta_bn / 2)
  }
}
