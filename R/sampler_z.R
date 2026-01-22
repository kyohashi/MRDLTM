#' Sample Latent Topic Assignments (z_cit)
#'
#' @description
#' Samples the latent topic assignment z_cit for each active observation.
#'
#' @param active_data A data frame of active observations (cust, item, time, y_cit).
#' @param state An environment containing the current MCMC state.
#' @param x_it A 3D array of marketing covariates (item, time, n_var).
#' @param n_obs Number of active observations.
#' @param n_topic Total number of topics (Z).
#' @param n_cust Total number of customers (C).
#' @param n_var Number of marketing covariates (M).
#' @param length_time Total length of time points (T).
#'
#' @return NULL
#' @importFrom stats pnorm
#' @noRd
sample_z = function(active_data, state, x_it, n_obs, n_topic, n_cust, n_var, length_time) {

  for (n in 1:n_obs) {
    c_idx = active_data$cust[n]
    i_idx = active_data$item[n]
    t_idx = active_data$time[n]
    y_val = active_data$y_cit[n]

    # --- 1. Topic Prior Weight (log-w_k) ---
    # In log-space: log(w_k) = eta for k < Z, and log(w_Z) = 0.
    # state$eta_zct is [Z-1, C, T]
    log_w = c(state$eta_zct[, c_idx, t_idx], 0)

    # --- 2. Probit Likelihood ---
    x_vec = x_it[i_idx, t_idx, ]
    xb = as.vector(state$beta_zi[, i_idx, ] %*% x_vec)

    # log(Phi(xb)) if y=1, log(1-Phi(xb)) if y=0.
    log_lik = pnorm(xb, lower.tail = (y_val == 1), log.p = TRUE)

    # --- 3. Posterior Sampling ---
    log_post = log_w + log_lik
    prob = exp(log_post - max(log_post))

    state$z_cit[n] = sample.int(n_topic, size = 1, prob = prob)
  }
}
