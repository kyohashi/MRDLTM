#' Sample DLM Variance Parameters (a2_z and b2_z)
#'
#' @description
#' Samples the observation variance (a2_z) and system variance (b2_z)
#' for the dynamic topic model.
#'
#' @param active_data A data frame of active observations (cust, item, time, y_cit).
#' @param state An environment containing the current MCMC state.
#' @param Dc Matrix of customer covariates (n_cust x p_dim).
#' @param n_topic Total number of topics (Z).
#' @param n_time Total number of time points (T).
#' @param n_cust Total number of customers (C).
#' @param p_dim Dimension of customer covariates (P).
#' @param priors A list of hyperparameters from the model object.
#'
#' @return NULL
#' @importFrom stats rgamma
#' @noRd
sample_dlm_vars = function(active_data, state, Dc, n_topic, n_time, n_cust, p_dim, priors) {
  # Default hyperparameters if not provided
  a2_shape = if (!is.null(priors$a2_shape)) priors$a2_shape else 0.01
  a2_scale = if (!is.null(priors$a2_scale)) priors$a2_scale else 0.01
  b2_shape = if (!is.null(priors$b2_shape)) priors$b2_shape else 0.01
  b2_scale = if (!is.null(priors$b2_scale)) priors$b2_scale else 0.01

  res = sample_dlm_vars_cpp(
    eta_zct_flat   = as.numeric(state$eta_zct),
    alpha_zt_flat  = as.numeric(state$alpha_zt),
    Dc_mat         = as.matrix(Dc),
    obs_cust     = as.integer(active_data$cust),
    obs_time     = as.integer(active_data$time),
    a2_prior_shape = a2_shape,
    a2_prior_scale = a2_scale,
    b2_prior_shape = b2_shape,
    b2_prior_scale = b2_scale,
    n_topic        = n_topic,
    n_time         = n_time,
    n_cust         = n_cust,
    p_dim          = p_dim
  )

  state$a2_z = res$a2_z
  state$b2_z = res$b2_z
}
