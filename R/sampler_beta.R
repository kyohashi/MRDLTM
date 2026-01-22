#' Sample Response Coefficients beta_zi
#'
#' @description
#' Sample the topic-item specific response coefficients beta_zi
#'
#' @param active_data A data frame of active observations (cust, item, time)
#' @param state An environment containing the current MCMC state
#' @param x_it An array of marketing covariates
#' @param n_item Number of items (I)
#' @param n_topic Number of latent topics (Z)
#' @param n_var Number of marketing covariates including intercept (M)
#' @param length_time Total length of time points (T).
#'
#' @return NULL
#' @noRd
sample_beta = function(active_data, state, x_it, n_item, n_topic, n_var, length_time){

  # Call optimized C++ sampler
  # Dimension check: x_it is [I, T, M]
  res_flat = sample_beta_cpp(
    z_cit        = as.integer(state$z_cit),
    item_idx     = as.integer(active_data$item),
    time_idx     = as.integer(active_data$time),
    u_cit        = as.numeric(state$u_cit),
    x_it_flat    = as.numeric(x_it),
    mu_i_mat     = as.matrix(state$mu_i),
    # Reorder V_i from [I, M, M] to [M, M, I] for efficient slicing in C++
    V_i_flat     = as.numeric(aperm(state$V_i, c(2, 3, 1))),
    n_topic      = n_topic,
    n_item       = n_item,
    n_time       = length_time,
    n_var        = n_var
  )

  # Reshape flattened result back to [Z, I, M]
  state$beta_zi = array(res_flat, dim = c(n_topic, n_item, n_var))
}
