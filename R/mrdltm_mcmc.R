#' Gibbs Sampler for MR-DLTM
#'
#' @description
#' Run the Gibbs sampler using a pre-defined mrdltm_model object.
#'
#' @param model An object of class "mrdltm_model".
#' @param iter Total MCMC iterations.
#' @param burnin Burn-in iterations.
#' @param quiet A flag to show a progress-bar
#'
#' @return A list of class "mrdltm_mcmc" containing the MCMC samples.
#' @export
mrdltm_mcmc = function(model, iter = 2000, burnin = 1000, quiet = TRUE) {

  # --- 1. Preparation ---
  obs = model$observations
  priors = model$priors

  ## Filter for active observations (C x I x T) -> (C_t x I_c x T_c)
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
    beta_zi   = array(0, dim = c(iter, n_topic, n_item, n_var)),
    mu_i   = array(0, dim = c(iter, n_item, n_var)),
    V_i    = array(0, dim = c(iter, n_item, n_var, n_var)),
    alpha_zt  = array(0, dim = c(iter, n_topic - 1, length_time, p_dim)),
    eta_zct   = array(0, dim = c(iter, n_topic - 1, n_cust, length_time)),
    z_cit     = array(0, dim = c(iter, n_obs)),
    a2_z   = matrix(0, nrow = iter, ncol = n_topic - 1),
    b2_z   = matrix(0, nrow = iter, ncol = n_topic - 1),
    log_lik = numeric(iter)
  )

  # --- 4. Gibbs Sampling Loop ---
  message(sprintf("Starting Gibbs Sampling: %d iterations (burn-in: %d)", iter, burnin))

  # Timers for progress display
  start_total <- proc.time()
  start_block <- proc.time()

  for (m in 1:iter) {

    # A. Topic Assignment
    sample_z(active_data, state, obs$x_it, n_item, n_topic, n_cust, n_var)

    # B. Latent Utility
    sample_u(active_data, state, obs$x_it, n_item, n_topic, n_cust, n_var)

    # C. Response Coefficients
    sample_beta(active_data, state, obs$x_it, n_item, n_topic, n_var, length_time)
    sample_mu_V(state, n_item, n_topic, n_var, obs$priors)

    # D. Topic Occupancy (DLM part)
    sample_eta(active_data, state, obs$Dc, n_cust, n_topic, length_time, p_dim)
    sample_alpha(active_data, state, obs$Dc, n_topic, length_time, p_dim, obs$priors)
    sample_dlm_vars(active_data, state, obs$Dc, n_topic, length_time, n_cust, p_dim, obs$priors)

    # --- 5. Record MCMC Samples ---
    history$beta_zi[m, , , ]  = state$beta_zi
    history$mu_i[m, , ]       = state$mu_i
    history$V_i[m, , , ]      = state$V_i
    history$alpha_zt[m, , , ] = state$alpha_zt
    history$eta_zct[m, , , ]  = state$eta_zct
    history$z_cit[m, ]        = state$z_cit
    history$a2_z[m, ]         = state$a2_z
    history$b2_z[m, ]         = state$b2_z
    history$log_lik[m]        = compute_log_likelihood(active_data, state, obs$x_it)

    # --- Progress message every 100 iterations ---
    if (!quiet && (m %% 100 == 0 || m == iter)) {
      current_time <- proc.time()
      elapsed_block <- (current_time - start_block)[3]
      elapsed_total <- (current_time - start_total)[3]

      phase <- if (m <= burnin) "Burn-in" else "Sampling"

      message(sprintf(
        "[%s] Iteration %d/%d - Last 100: %.2fs (Total: %.1fs)",
        phase, m, iter, elapsed_block, elapsed_total
      ))

      # Reset block timer
      start_block <- current_time
    }
  }

  # Assign class for future S3 methods (bayesplot conversion, etc.)
  class(history) = "mrdltm_mcmc"

  return(history)
}
