#' Sample Inner state alpha_zt using FFBS
#'
#' @description
#' Samples the dynamic state coefficients for the topic occupancy model
#' using Forward Filtering Backward Sampling.
#'
#' @param state An environment containing the current MCMC state.
#' @param Dc Matrix of customer covariates (n_cust x p_dim).
#' @param n_topic Total number of topics (Z).
#' @param length_time Total length of time points (T).
#' @param p_dim Dimension of customer covariates (P).
#' @param priors A list of hyperparameters from the model object.
#'
#' @return NULL
#' @importFrom MASS mvrnorm
#' @noRd
sample_alpha = function(state, Dc, n_topic, length_time, p_dim, priors) {
  n_z_dlm = n_topic - 1

  # --- 1. Hyperparameters for Initial State (alpha_z0) ---
  # mz0: Prior mean for alpha_z0. Default: Zero vector.
  mz0 = if (!is.null(priors$mz0)) priors$mz0 else rep(0, p_dim)

  # Sz0: Prior covariance for alpha_z0. Default: 10 * Identity (Diffuse).
  Sz0 = if (!is.null(priors$Sz0)) priors$Sz0 else diag(10, p_dim)

  # Precompute constant D'D part for observation precision
  DtD = t(Dc) %*% Dc

  for (k in 1:n_z_dlm) {
    # Topic-specific variances
    b2 = state$b2_z[k]
    a2 = state$a2_z[k]

    # Storage for filtering results
    m_filt = matrix(0, nrow = length_time, ncol = p_dim)
    C_filt = array(0, dim = c(length_time, p_dim, p_dim))

    # --- 2. Forward Filtering ---
    m_curr = mz0
    C_curr = Sz0

    for (t in 1:length_time) {
      # Time Update (Prediction)
      # alpha_t = alpha_{t-1} + noise (G = Identity matrix for Random Walk)
      m_pred = m_curr
      R_pred = C_curr + diag(b2, p_dim)
      R_inv = solve(R_pred)

      # Observation Update (Filtering)
      # Precision Q = R^-1 + (D'D / a2)
      # Mean m = Q^-1 * (R^-1 * m_pred + (D'eta) / a2)
      eta_t = state$eta_zct[k, , t]

      info_prec = DtD / a2
      info_mean = as.vector(t(Dc) %*% eta_t) / a2

      C_curr_inv = R_inv + info_prec
      C_curr = solve(C_curr_inv)
      m_curr = C_curr %*% (R_inv %*% m_pred + info_mean)

      m_filt[t, ] = m_curr
      C_filt[t, , ] = C_curr
    }

    # --- 3. Backward Sampling ---
    # Final state sampling: alpha_T ~ N(m_T, C_T)
    state$alpha_zt[k, length_time, ] = mvrnorm(1, m_filt[length_time, ], C_filt[length_time, , ])

    # Recursively sample backward: alpha_t | alpha_{t+1}
    for (t in (length_time - 1):1) {
      W_mat = diag(b2, p_dim)
      # Smoothing gain: B_t = C_t * (C_t + W)^{-1}
      # Using solve for C_t + W
      gain = C_filt[t, , ] %*% solve(C_filt[t, , ] + W_mat)

      m_back = m_filt[t, ] + as.vector(gain %*% (state$alpha_zt[k, t + 1, ] - m_filt[t, ]))
      V_back = C_filt[t, , ] - gain %*% C_filt[t, , ]

      # Ensure symmetry for numerical stability
      V_back = (V_back + t(V_back)) / 2

      state$alpha_zt[k, t, ] = mvrnorm(1, m_back, V_back)
    }
  }
}
