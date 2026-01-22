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
#'
#' @return NULL
#' @noRd
sample_beta = function(active_data, state, x_it, n_item, n_topic, n_var){

  mu_beta_zi <- as.matrix(state$mu_i)

  V_inv_izi_array <- array(0, dim = c(n_var, n_var, n_item))
  for (i in 1:n_item) {
    V_i_mat <- state$V_i[i, , ]
    if (is.null(dim(V_i_mat))) {
      V_i_mat <- matrix(V_i_mat, n_var, n_var)
    }
    V_inv_izi_array[, , i] <- solve(V_i_mat)
  }

  state$beta_zi = sample_beta_cpp(
    u_cit = state$u_cit,
    x_it_matrix = matrix(x_it, ncol = n_var),
    z_cit = state$z_cit,
    item_idx = active_data$item,
    time_idx = active_data$time,
    mu_beta_zi = mu_beta_zi,
    V_inv_zi = as.numeric(V_inv_izi_array),
    n_topic = n_topic,
    n_item = n_item,
    n_var = n_var
  )
}
