#' Gibbs Sampler for MR-DLTM
#'
#' @description
#' Run the Gibbs sampler using a pre-defined mrdltm_model object.
#'
#' @param model An object of class "mrdltm_model".
#' @param iter Total MCMC iterations.
#' @param burnin Burn-in iterations.
#'
#' @return A list of class "mrdltm_mcmc" containing the MCMC samples.
#' @export
mrdltm_mcmc = function(model, iter = 2000, burnin = 1000) {

  # --- 1. Preparation ---
  obs = model$observations
  priors = model$priors

  ## Filter for active observations
  active_data = filter_active_data(obs$data)

  ## Get dimensions
  n_obs       = nrow(active_data)
  n_item      = dim(obs$x_it)[1]
  length_time = dim(obs$x_it)[2]
  n_var       = dim(obs$x_it)[3]
  n_cust      = nrow(obs$Dc)
  n_topic     = model$n_topic
  p_dim       = model$p_dim

  # --- 2. State initialization ---
  state = init_state(
    active_data = active_data,
    n_item      = n_item,
    n_cust      = n_cust,
    n_topic     = n_topic,
    length_time = length_time,
    n_var       = n_var,
    p_dim       = p_dim
  )

  # --- 3. Pre-allocate history ---
  # We store all iterations to allow users to inspect burn-in if needed.
  history = list(
    beta   = array(0, dim = c(iter, n_topic, n_item, n_var)),
    mu_i   = array(0, dim = c(iter, n_item, n_var)),
    V_i    = array(0, dim = c(iter, n_item, n_var, n_var)),
    alpha  = array(0, dim = c(iter, n_topic - 1, length_time, p_dim)),
    a2_z   = matrix(0, nrow = iter, ncol = n_topic - 1),
    b2_z   = matrix(0, nrow = iter, ncol = n_topic - 1),
    log_lik = numeric(iter)
  )

  # --- 4. Gibbs Sampling Loop ---
  message(sprintf("Starting Gibbs Sampling: %d iterations (burn-in: %d)", iter, burnin))

  for (m in 1:iter) {

    # A. Topic Assignment (Marginalized sample_z should come first for better mixing)
    sample_z(active_data, state, obs$x_it, n_item, n_topic, n_cust, n_var)

    # B. Latent Utility (Probit part)
    sample_u(active_data, state, obs$x_it, n_item, n_topic, n_var)

    # C. Response Coefficients (Hierarchical part)
    sample_beta(active_data, state, obs$x_it, n_item, n_topic, n_var)
    sample_mu_V(state, n_item, n_topic, n_var, obs$priors)

    # D. Topic Occupancy (DLM part)
    sample_eta(active_data, state, obs$Dc, n_cust, n_topic, length_time)
    sample_alpha(state, obs$Dc, n_topic, length_time, p_dim, obs$priors)
    sample_dlm_vars(state, obs$Dc, n_topic, length_time, n_cust, p_dim, obs$priors)

    # --- 5. Record MCMC Samples ---
    history$beta[m, , , ]  = state$beta_zi
    history$mu_i[m, , ]    = state$mu_i
    history$V_i[m, , , ]   = state$V_i
    history$alpha[m, , , ] = state$alpha_zt
    history$a2_z[m, ]      = state$a2_z
    history$b2_z[m, ]      = state$b2_z

    # Calculate Log-Likelihood (for convergence diagnostic)
    # Using the marginalized probit likelihood for the current topics
    history$log_lik[m] = compute_log_likelihood(active_data, state, obs$x_it)

    # Progress message
    if (m %% 100 == 0) {
      status = ifelse(m <= burnin, "(Burn-in)", "(Sampling)")
      message(sprintf("Iteration %d / %d %s", m, iter, status))
    }
  }

  # Assign class for future S3 methods (bayesplot conversion, etc.)
  class(history) = "mrdltm_mcmc"

  return(history)
}

#' Helper to compute log-likelihood for the current state
#' @noRd
compute_log_likelihood = function(active_data, state, x_it) {
  # Marginalized over u: Log Phi(xb) or Log(1-Phi(xb))
  # Summed over all active observations
  n_obs = nrow(active_data)
  log_lik_vec = numeric(n_obs)

  # This part can be vectorized if performance becomes an issue
  for (n in 1:n_obs) {
    z = state$z_cit[n]
    i = active_data$item[n]
    t = active_data$time[n]
    y = active_data$y_cit[n]

    xb = sum(x_it[i, t, ] * state$beta_zi[z, i, ])
    log_lik_vec[n] = stats::pnorm(xb, lower.tail = (y == 1), log.p = TRUE)
  }

  return(sum(log_lik_vec))
}
