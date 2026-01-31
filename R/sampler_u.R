#' Sample Utility
#'
#' @description
#' Sample latent utility from a truncated normal distribution
#' The result is updated directly within the state environment.
#'
#' @param active_data A data frame of active observations (c, i, t, y) in C x Ic x Tc.
#' @param state An environment containing the current MCMC state.
#' @param x_it An array of marketing covariates.
#' @param n_item Total number of items (I).
#' @param n_topic Total number of topics (Z).
#' @param n_cust Total number of customers (C).
#' @param n_var Number of marketing covariates (M).
#'
#' @return NULL
#' @noRd
sample_u = function(active_data, state, x_it, n_item, n_topic, n_cust, n_var){

  # --- Prepare memory layout for C++ matrix-vector operations ---
  # beta_zi: [Z, I, M] -> [Z, M, I] to make slice(i) a [Z x M] matrix
  beta_prep = aperm(state$beta_zi, c(1, 3, 2))

  # x_it: [I, T, M] -> [M, T, I] to make slice(i).col(t) an M-vector
  x_prep = aperm(x_it, c(3, 2, 1))

  state$u_cit = sample_u_cpp(
    y_cit        = active_data$y_cit,
    beta_flat    = as.numeric(beta_prep),
    x_flat       = as.numeric(x_prep),
    z_cit        = state$z_cit,
    item_idx     = active_data$item,
    time_idx     = active_data$time,
    n_topic      = n_topic,
    n_item       = n_item,
    n_time       = dim(x_it)[2],
    n_cust       = n_cust,
    n_var        = n_var
  )
}
