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

  t_z = 0
  t_u = 0
  t_beta = 0
  t_muV = 0
  t_eta = 0
  t_alpha = 0
  t_vars = 0
  t_lik = 0

  for (m in 1:iter) {

    # A. Topic Assignment (Marginalized sample_z should come first for better mixing)
    t <- proc.time()
    sample_z(active_data, state, obs$x_it, n_item, n_topic, n_cust, n_var)
    t_z <- t_z + (proc.time() - t)[3]

    # B. Latent Utility (Probit part)
    t <- proc.time()
    sample_u(active_data, state, obs$x_it, n_item, n_topic, n_var)
    t_u <- t_u + (proc.time() - t)[3]

    # C. Response Coefficients (Hierarchical part)
    t <- proc.time()
    sample_beta(active_data, state, obs$x_it, n_item, n_topic, n_var)
    t_beta <- t_beta + (proc.time() - t)[3]

    t <- proc.time()
    sample_mu_V(state, n_item, n_topic, n_var, obs$priors)
    t_muV <- t_muV + (proc.time() - t)[3]

    # D. Topic Occupancy (DLM part)
    t <- proc.time()
    sample_eta(active_data, state, obs$Dc, n_cust, n_topic, length_time)
    t_eta <- t_eta + (proc.time() - t)[3]

    t <- proc.time()
    sample_alpha(state, obs$Dc, n_topic, length_time, p_dim, obs$priors)
    t_alpha <- t_alpha + (proc.time() - t)[3]

    t <- proc.time()
    sample_dlm_vars(state, obs$Dc, n_topic, length_time, n_cust, p_dim, obs$priors)
    t_vars <- t_vars + (proc.time() - t)[3]

    # --- 5. Record MCMC Samples ---
    history$beta[m, , , ]  = state$beta_zi
    history$mu_i[m, , ]    = state$mu_i
    history$V_i[m, , , ]   = state$V_i
    history$alpha[m, , , ] = state$alpha_zt
    history$a2_z[m, ]      = state$a2_z
    history$b2_z[m, ]      = state$b2_z

    # Calculate Log-Likelihood (for convergence diagnostic)
    # Using the marginalized probit likelihood for the current topics
    t <- proc.time()
    history$log_lik[m] = compute_log_likelihood(active_data, state, obs$x_it)
    t_lik = t_lik + (proc.time() - t)[3]

    # Progress message
    if (m %% 100 == 0) {
      status = ifelse(m <= burnin, "(Burn-in)", "(Sampling)")
      message(sprintf("Iteration %d / %d %s", m, iter, status))
      message(sprintf("\nIter %d times - \n Z: %.3fs \n U: %.3fs \n Beta: %.3fs \n muV: %.3fs \n Eta: %.3fs \n Alpha: %.3fs \n Vars: %.3fs \n Lik: %.3fs", m, t_z, t_u, t_beta, t_muV, t_eta, t_alpha, t_vars, t_lik))
    }
  }

  # Assign class for future S3 methods (bayesplot conversion, etc.)
  class(history) = "mrdltm_mcmc"

  return(history)
}

#' Helper to compute log-likelihood for the current state
#' @noRd
compute_log_likelihood = function(active_data, state, x_it) {
  # Extract dimensions from the state/data
  n_topic = dim(state$beta_zi)[1]
  n_item  = dim(state$beta_zi)[2]
  n_var   = dim(state$beta_zi)[3]
  n_time  = dim(x_it)[2]

  log_lik = compute_log_likelihood_cpp(
    z_cit        = as.integer(state$z_cit),
    item_idx     = as.integer(active_data$item),
    time_idx     = as.integer(active_data$time),
    y_cit        = as.integer(active_data$y_cit),
    beta_zi_flat = as.numeric(state$beta_zi),
    x_it_flat    = as.numeric(x_it),
    n_topic      = n_topic,
    n_item       = n_item,
    n_time       = n_time,
    n_var        = n_var
  )

  return(log_lik)
}
