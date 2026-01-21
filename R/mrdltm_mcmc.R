#' Gibbs Sampler for MR-DLTM
#'
#' @description
#' Run the Gibbs sampler using a pre-defined mrdltm_model object.
#'
#' @param model An object of class "mrdltm_model".
#' @param iter Total mcmc iterations
#' @param burnin Burn-in iterations
#'
#' @return A list containing the mcmc samples
#' @export
mrdltm_mcmc = function(model, iter = 2000, burnin = 1000){

  # --- Preparation ---
  obs = model$observations

  ## filter for active observations (Ic and Tc)
  active_data = filter_active_data(obs$data)

  ## Get dimensions
  n_item = dim(obs$x_it)[1]
  length_time = dim(obs$x_it)[2]
  n_var = dim(obs$x_it)[3]
  n_cust = nrow(obs$Dc)

  # --- State initialization ---
  state = init_state(
    active_data = active_data,
    n_item = n_item,
    n_cust = n_cust,
    n_topic = model$n_topic,
    length_time = length_time,
    n_var = n_var,
    p_dim = model$p_dim
  )

  # Pre-allocate history for ALL iterations
  history = list(
    beta = array(0, dim = c(iter, model$n_topic, n_item, n_var)),
    alpha = array(0, dim = c(iter, model$n_topic - 1, length_time, model$p_dim))
    # other params to be stored
  )

  # --- Gibbs Sampling Loop ---
  for (m in 1:iter){
    # Each sampler updates 'state' environment in-place
    sample_u(active_data, state, obs$x_it)
    sample_z(active_data, state)
    sample_beta(active_data, state, obs$x_it)
    sample_mu_V(state)
    sample_eta_omega(state)

    # Pass Gt and the bound Dc (customer covariates)
    sample_alpha(state, model$Gt, obs$Dc)

    sample_a2_b2(state)

    # --- 4. Record Samples (Every iteration) ---
    history$beta[m, , , ] <- state$beta_zi
    history$alpha[m, , , ] <- state$alpha_zt

    if (m %% 100 == 0) {
      status <- ifelse(m <= burnin, "(Burn-in)", "(Sampling)")
      message(sprintf("Iteration %d / %d %s", m, iter, status))
    }
  }

  return(history)
}




