#' Sample Response Coefficients beta_zi
#'
#' @description
#' Sample the topic-item specific response coefficients beta_zi
#'
#' @param active_data A data frame of active observations (c, i, t)
#' @param state An environment containing the current MCMC state
#' @param x_it An array of marketing covariates
#'
#' @return NULL
#' @importFrom MASS mvrnorm
#' @noRd
sample_beta = function(active_data, state, x_it){
  n_topic = dim(state$beta_zi)[1]
  n_item = dim(state$beta_zi)[2]
  n_var = dim(state$beta_zi)[3]

  # Loop through (z, i) combination
  for (z in 1:n_topic){
    for (i in 1:n_item){
      # identify corresponding indices for (z, i)
      idx = which(active_data$item == i & state$z_cit == z)

      # --- get required statistics ---
      ## priors
      mu_i = state$mu_i[i, ]
      V_i_inv = solve(state$V_i[i,,])

      if (length(idx) == 0) {
        # If no observations, sample from the hierarchical prior
        state$beta_zi[z, i, ] = as.vector(
          mvrnorm(1, mu = mu_i, Sigma = state$V_i[i, , ])
        )
        next
      }

      ## utility
      u_zi = state$u_cit[idx]

      ## X_zi
      times = active_data$time[idx]
      X_zi = matrix(x_it[i, times, ], nrow = length(idx), ncol = n_var)

      # --- posterior parameters ---
      R = t(X_zi) %*% X_zi + V_i_inv
      R_inv = solve(R)
      post_mean = R_inv %*% (t(X_zi) %*% u_zi + V_i_inv %*% mu_i)

      # --- sample from poserior ---
      state$beta_zi[z, i, ] = as.vector(
        mvrnorm(1, mu = post_mean, Sigma = R_inv)
      )
    }
  }
}
