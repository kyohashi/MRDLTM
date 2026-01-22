#' Sample Latent Topic Assignments (z_cit)
#'
#' @description
#' Samples the latent topic assignment z_cit for each active observation.
#'
#' @param active_data A data frame of active observations (cust, item, time, y_cit).
#' @param state An environment containing the current MCMC state.
#' @param x_it A 3D array of marketing covariates (item, time, n_var).
#' @param n_item Total number of items (I).
#' @param n_topic Total number of topics (Z).
#' @param n_cust Total number of customers (C).
#' @param n_var Number of marketing covariates (M).
#'
#' @return NULL
#' @noRd
sample_z = function(active_data, state, x_it, n_item, n_topic, n_cust, n_var) {

  state$z_cit = sample_z_cpp(
    u_cit        = state$u_cit,
    eta_zct_flat = as.numeric(state$eta_zct),
    beta_zi_flat = as.numeric(state$beta_zi),
    x_it_matrix  = matrix(x_it, ncol = n_var),
    cust_idx     = active_data$cust,
    item_idx     = active_data$item,
    time_idx     = active_data$time,
    n_topic      = n_topic,
    n_item       = n_item,
    n_cust       = n_cust,
    n_var        = n_var
  )
}
