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
sample_mu_V = function(state, n_item, n_topic, n_var, priors) {

  # --- Hyperparameters ---
  mu_tilde       = if (!is.null(priors$mu_tilde)) priors$mu_tilde else rep(0, n_var)
  sigma_tilde_mu = if (!is.null(priors$sigma_tilde_mu)) priors$sigma_tilde_mu else 10.0
  w_tilde        = if (!is.null(priors$w_tilde)) priors$w_tilde else (n_var + 2)
  W_tilde        = if (!is.null(priors$W_tilde)) priors$W_tilde else diag(1, n_var)

  # Dimension safety check
  mu_tilde = as.numeric(mu_tilde)
  if(length(mu_tilde) != n_var) mu_tilde = rep(mu_tilde[1], n_var)
  W_tilde = as.matrix(W_tilde)
  if(nrow(W_tilde) != n_var) W_tilde = diag(as.numeric(W_tilde[1]), n_var)

  # --- Prepare memory layout for C++ ---
  # Move 'item' to the 3rd dimension: [topic, var, item]
  beta_prepared = aperm(state$beta_zi, c(1, 3, 2))
  # Move 'item' to the 3rd dimension: [var, var, item]
  V_prepared = aperm(state$V_i, c(2, 3, 1))

  # --- Call C++ Sampler ---
  # Names here must match exactly with the 'arma::' prefixed arguments in C++
  res = sample_mu_V_cpp(
    beta_zi_flat   = as.numeric(beta_prepared),
    mu_i_mat       = as.matrix(state$mu_i),
    V_i_flat       = as.numeric(V_prepared),
    mu_tilde       = mu_tilde,       # Changed from mu_tilde_vec
    sigma_tilde_mu = sigma_tilde_mu,
    w_tilde        = w_tilde,
    W_tilde        = W_tilde,        # Changed from W_tilde_mat
    n_topic        = n_topic,
    n_item         = n_item,
    n_var          = n_var
  )

  # --- Restore state dimensions ---
  state$mu_i = res$mu_i
  state$V_i  = aperm(res$V_i, c(3, 1, 2))
}
