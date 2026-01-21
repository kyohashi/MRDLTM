#' Sample Utility
#'
#' @description
#' Sample latent utility from a truncated normal distribution
#'
#' @param active_data A data frame of active observations which means (c,i,t) combo in C x Ic x Tc
#' @param state A list of current parameters (beta_zi, z_cit)
#' @param x_it An array of marketing covariates
#'
#' @return A numeric vector of sampled utilities
#' @importFrom truncnorm rtruncnorm
#' @noRd
sample_u = function(active_data, state, x_it){

  # --- Calculate the means of utilities ---
  ## u_cit ~ N(beta_zi * x_it, 1)
  u_means = sapply(1:nrow(active_data), function(idx){
    z = state$z_cit[idx]
    i = active_data$item[idx]
    t = active_data$time[idx]
    sum(state$beta_zi[z, i, ] * x_it[i, t, ])
  })

  # --- Sample from truncated normal ---
  u_new = truncnorm::rtruncnorm(
    n = nrow(active_data),
    a = ifelse(active_data$y_cit == 1, 0, -Inf), # lower bounds
    b = ifelse(active_data$y_cit == 1, Inf, 0),  # upper bounds
    mean = u_means,
    sd = 1
  )

  return(u_new)
}
