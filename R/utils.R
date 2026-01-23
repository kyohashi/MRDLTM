#' Initialize MCMC State
#'
#' @description
#' Creates the initial states required for the MCMC(Gibbs Sampler)
#'
#' @param active_data A data frame of active observations which means (c,i,t) combo in C x Ic x Tc
#' @param n_item Number of items (I)
#' @param n_cust Number of customers (C)
#' @param n_topic Number of latent topics (Z)
#' @param length_time Length of time points (T)
#' @param n_var Number of marketing covariates (M)
#' @param p_dim Dimension of the DLM staete vector (alpha_zt)
#'
#' @return An environment containing initial values
#' @noRd
init_state = function(active_data, n_item, n_cust, n_topic, length_time, n_var, p_dim){

  # Create a new environment for the state.
  # This avoids the overhead of list copying in every MCMC iteration.
  state = new.env(parent = emptyenv())

  # --- utility (u_cit) ---
  state$u_cit = ifelse(active_data$y_cit > 0, 0.5, -0.5)

  # --- latent topic assignments (z_cit) ----
  state$z_cit = sample(1:n_topic, nrow(active_data), replace = TRUE)

  # --- response coef. (beta_zi) ---
  state$beta_zi = array(0, dim = c(n_topic, n_item, n_var))

  # --- Prior params of beta ---
  state$mu_i = matrix(0, nrow = n_item, ncol = n_var)
  state$V_i = array(0, dim = c(n_item, n_var, n_var))
  for (i in 1:n_item){
    state$V_i[i,,] = diag(1, n_var)
  }

  # --- DLM part ---
  n_z_dlm = n_topic - 1 # for identifiability

  # --- latent obs. (eta_zct) ---
  state$eta_zct = array(0, dim = c(n_z_dlm, n_cust, length_time))

  # --- inner state (alpha_zt) ---
  state$alpha_zt = array(0, dim = c(n_z_dlm, length_time, p_dim))

  # --- DLM variances ---
  state$a2_z = rep(1.0, n_z_dlm) # obs variance
  state$b2_z = rep(0.1, n_z_dlm) # system variance

  # --- Polya-Gamma parameters ---
  state$omega_zct = array(1, dim = c(n_z_dlm, n_cust, length_time))
  state$kappa_zct = array(0, dim = c(n_z_dlm, n_cust, length_time))

  return(state)
}

#' Filter for Active Observations (Ic and Tc)
#'
#' @description
#' Filters the data to include only observations where:
#' 1. The item i belongs to Ic (items purchased by customer c at least once).
#' 2. The time t belongs to Tc (time points where customer c made at least one purchase).
#'
#' @param data A data frame containing (cust, item, time, y_cit)
#'
#' @return A filtered data frame of active observations.
#' @importFrom dplyr group_by filter mutate ungroup select semi_join
#' @noRd
filter_active_data = function(data) {

  # Find the set of items Ic for each customer
  ic_set = data |>
    group_by(cust, item) |>
    filter(sum(y_cit) > 0) |>
    ungroup() |>
    select(cust, item) |>
    unique()

  # Find the set of time points Tc for each customer
  tc_set = data |>
    group_by(cust, time) |>
    filter(sum(y_cit) > 0) |>
    ungroup() |>
    select(cust, time) |>
    unique()

  # Filter the original data
  # Keep rows where (cust, item) is in Ic AND (cust, time) is in Tc
  active_data = data |>
    semi_join(ic_set, by = c("cust", "item")) |>
    semi_join(tc_set, by = c("cust", "time"))

  return(active_data)
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

#' Convert MCMC samples to array for bayesplot
#'
#' @param x An object of class "mrdltm_mcmc".
#' @param parameter Name of the parameter group to extract.
#' @param ... Not used.
#'
#' @return A 3D array (iteration, chain, parameter)
#' @export
as.array.mrdltm_mcmc <- function(x, parameter = "log_lik", ...) {
  samples <- x[[parameter]]
  if (is.null(samples)) stop(paste("Parameter", parameter, "not found."))

  # --- Case: log_lik (Vector [iter]) ---
  if (parameter == "log_lik") {
    out <- array(samples, dim = c(length(samples), 1, 1))
    dimnames(out) <- list(NULL, "chain:1", "log_lik")
    return(out)
  }

  # --- Case: Variance params (Matrix [iter, n_topic-1]) ---
  if (parameter %in% c("a2_z", "b2_z")) {
    samples_mat <- as.matrix(samples)
    out <- array(samples_mat, dim = c(nrow(samples_mat), 1, ncol(samples_mat)))
    dimnames(out) <- list(NULL, "chain:1", paste0(parameter, "[", 1:ncol(samples_mat), "]"))
    return(out)
  }

  # --- Case: Multi-dim arrays (beta, alpha, mu_i, V_i) ---
  dims <- dim(samples)
  n_iter <- dims[1]
  total_params <- prod(dims[-1])

  out <- array(samples, dim = c(n_iter, 1, total_params))

  # Generate labels based on dimension structure
  if (parameter == "beta") {
    # beta [iter, topic, item, var]
    p_names <- expand.grid(v = 1:dims[4], i = 1:dims[3], z = 1:dims[2])
    names_vec <- apply(p_names, 1, function(p) sprintf("beta[z%d,i%d,v%d]", p[3], p[2], p[1]))
  } else if (parameter == "alpha") {
    # alpha [iter, topic-1, time, p_dim]
    p_names <- expand.grid(p = 1:dims[4], t = 1:dims[3], z = 1:dims[2])
    names_vec <- apply(p_names, 1, function(p) sprintf("alpha[z%d,t%d,p%d]", p[3], p[2], p[1]))
  } else {
    names_vec <- paste0(parameter, "[", 1:total_params, "]")
  }

  dimnames(out) <- list(NULL, "chain:1", names_vec)
  return(out)
}

#' Extract MCMC Samples with Burn-in and Thinning
#'
#' @param x An object of class "mrdltm_mcmc".
#' @param parameter Name of the parameter group.
#' @param burnin Number of iterations to discard. Defaults to 0.
#' @param thin Interval for thinning. Defaults to 1.
#'
#' @return A 3D array (iteration, chain, parameter)
#' @export
extract_samples <- function(x, parameter = "log_lik", burnin = 0, thin = 1) {
  # Call the S3 method as.array to get the full array
  arr <- as.array(x, parameter = parameter)

  iter_total <- dim(arr)[1]

  # Selection logic
  indices <- seq(from = burnin + 1, to = iter_total, by = thin)

  if (length(indices) == 0) {
    stop("No samples left after applying burn-in and thinning.")
  }

  # Subset along the first dimension (Iterations)
  return(arr[indices, , , drop = FALSE])
}
