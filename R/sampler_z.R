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
#' @importFrom stats runif
#' @noRd
sample_z = function(active_data, state, x_it, n_item, n_topic, n_cust, n_var) {

  # --- Prepare memory layout for C++ matrix-vector operations ---
  # beta_zi: [Z, I, M] -> [Z, M, I] to make slice(i) a [Z x M] matrix
  beta_prep = aperm(state$beta_zi, c(1, 3, 2))

  # x_it: [I, T, M] -> [M, T, I] to make slice(i).col(t) an M-vector
  x_prep = aperm(x_it, c(3, 2, 1))

  # Pre-generate random uniforms to avoid non-thread-safe R API calls in parallel loop
  n_obs = length(state$u_cit)
  rand_u = stats::runif(n_obs)

  state$z_cit = sample_z_cpp(
    y_cit        = active_data$y_cit,
    eta_flat     = as.numeric(state$eta_zct),
    beta_flat    = as.numeric(beta_prep),
    x_flat       = as.numeric(x_prep),
    cust_idx     = as.integer(active_data$cust),
    item_idx     = as.integer(active_data$item),
    time_idx     = as.integer(active_data$time),
    rand_u       = rand_u,
    n_topic      = n_topic,
    n_item       = n_item,
    n_time       = dim(x_it)[2],
    n_cust       = n_cust,
    n_var        = n_var
  )
}
