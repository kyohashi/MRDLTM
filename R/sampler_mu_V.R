#' Sample Hierarchical Priors for beta (mu_i and V_i)
#'
#' @description
#' Samples the item-specific hierarchical mean (mu_i) and variance (V_i)
#'
#' @param state An environment containing the current MCMC state.
#' @param priors A list of hyperparameters from the model object.
#' @param n_item Number of items (I)
#' @param n_topic Number of latent topics (Z)
#' @param n_var Number of marketing covariates including intercept (M)
#'
#' @return NULL
#' @importFrom MASS mvrnorm
#' @importFrom MCMCpack riwish
#' @noRd
sample_mu_V = function(state, priors, n_item, n_topic, n_var) {

  # --- Hyperparameters ---
  mu_tilde = if (!is.null(priors$mu_tilde)) priors$mu_tilde else rep(0, n_var)
  sigma_tilde_mu = if (!is.null(priors$sigma_tilde_mu)) priors$sigma_tilde_mu else 10.0
  w_tilde = if (!is.null(priors$w_tilde)) priors$w_tilde else (n_var + 2)
  W_tilde = if (!is.null(priors$W_tilde)) priors$W_tilde else diag(1, n_var)

  for (i in 1:n_item) {
    # Extract betas for item i: [n_topic x n_var]
    beta_i = matrix(state$beta_zi[, i, ], nrow = n_topic, ncol = n_var)
    beta_bar = colMeans(beta_i)

    # Sum of squared deviations from the sample mean
    S_data = matrix(0, n_var, n_var)
    for (z in 1:n_topic) {
      diff_b = beta_i[z, ] - beta_bar
      S_data = S_data + diff_b %*% t(diff_b)
    }

    # --- Sample V_i from Marginal Posterior ---
    post_w = w_tilde + n_topic
    post_W = W_tilde + S_data
    state$V_i[i, , ] = riwish(v = post_w, S = post_W)

    # --- Sample mu_i from Conditional Posterior ---
    prec_scale = (1 / sigma_tilde_mu) + n_topic
    post_mu_cov = state$V_i[i, , ] / prec_scale

    # post_mu_mean = (mu_tilde / sigma_tilde_mu + Z * beta_bar) / prec_scale
    post_mu_mean = (mu_tilde / sigma_tilde_mu + n_topic * beta_bar) / prec_scale

    state$mu_i[i, ] = as.vector(
      mvrnorm(1, mu = post_mu_mean, Sigma = post_mu_cov)
    )
  }
}
