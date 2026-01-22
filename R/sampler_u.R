#' Sample Utility
#'
#' @description
#' Sample latent utility from a truncated normal distribution
#' The result is updated directly within the state environment.
#'
#' @param active_data A data frame of active observations which means (c,i,t, y) combo in C x Ic x Tc
#' @param state An environment containing the current MCMC state
#' @param x_it An array of marketing covariates
#' @param n_item Total number of items (I).
#' @param n_topic Total number of topics (Z).
#' @param n_var Number of marketing covariates (M).
#'
#' @return NULL
#' @noRd
sample_u = function(active_data, state, x_it, n_item, n_topic, n_var){

  state$u_cit = sample_u_cpp(
    y_cit        = active_data$y_cit,
    x_it_matrix  = matrix(x_it, ncol = n_var),
    beta_zi_flat = as.numeric(state$beta_zi),
    z_cit        = state$z_cit,
    item_idx     = active_data$item,
    time_idx     = active_data$time,
    n_topic      = n_topic,
    n_item       = n_item,
    n_var        = n_var
  )
}
