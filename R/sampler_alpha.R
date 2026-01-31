#' Sample Inner state alpha_zt using FFBS
#'
#' @description
#' Samples the dynamic state coefficients for the topic occupancy model
#' using Forward Filtering Backward Sampling.
#'
#' @param active_data A data frame of active observations (cust, item, time, y_cit).
#' @param state An environment containing the current MCMC state.
#' @param Dc Matrix of customer covariates (n_cust x p_dim).
#' @param n_topic Total number of topics (Z).
#' @param n_time Total number of time points (T).
#' @param p_dim Dimension of customer covariates (P).
#' @param priors A list of hyperparameters from the model object.
#'
#' @return NULL
#' @noRd
sample_alpha = function(active_data, state, Dc, n_topic, n_time, p_dim, priors) {
  mz0 = if (!is.null(priors$mz0)) priors$mz0 else rep(0, p_dim)
  Sz0 = if (!is.null(priors$Sz0)) priors$Sz0 else diag(10, p_dim)

  state$alpha_zt = sample_alpha_cpp(
    eta_zct_flat = as.numeric(state$eta_zct),
    Dc_mat       = Dc,
    a2_z         = state$a2_z,
    b2_z         = state$b2_z,
    mz0_vec      = mz0,
    Sz0_mat      = Sz0,
    obs_cust     = as.integer(active_data$cust),
    obs_time     = as.integer(active_data$time),
    n_topic      = n_topic,
    n_time       = n_time,
    n_cust       = nrow(Dc),
    p_dim        = p_dim
  )
}
