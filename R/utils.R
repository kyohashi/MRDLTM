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

#' Convert MCMC samples to array for bayesplot
#'
#' @param x An object of class "mrdltm_mcmc".
#' @param parameter Name of the parameter group to extract.
#' @param ... Not used.
#'
#' @return A 3D array [iteration, chain, parameter]
#' @export
as.array.mrdltm_mcmc = function(x, parameter = "log_lik", ...) {
  samples = x[[parameter]]

  if (is.null(samples)) stop(paste("Parameter", parameter, "not found in results."))

  # --- Case: log_lik (Vector [iter]) ---
  if (parameter == "log_lik") {
    out = array(samples, dim = c(length(samples), 1, 1))
    dimnames(out) = list(NULL, "chain:1", "log_lik")
    return(out)
  }

  # --- Case: Variance params (Matrix [iter, n_z_dlm]) ---
  if (parameter %in% c("a2_z", "b2_z")) {
    n_z = ncol(samples)
    out = array(samples, dim = c(nrow(samples), 1, n_z))
    dimnames(out) = list(NULL, "chain:1", paste0(parameter, "[", 1:n_z, "]"))
    return(out)
  }

  # --- Case: multi-dim arrays (beta, alpha, mu_i, V_i) ---
  dims = dim(samples)
  n_iter = dims[1]
  total_params = prod(dims[2:length(dims)])

  # Flatten other dimensions into one parameter dimension
  out = array(samples, dim = c(n_iter, 1, total_params))

  # Generate labels based on dimension structure
  if (parameter == "beta") {
    # beta [iter, topic, item, var]
    p_names = expand.grid(v = 1:dims[4], i = 1:dims[3], z = 1:dims[2])
    names_vec = apply(p_names, 1, function(p) sprintf("beta[z%d,i%d,v%d]", p[3], p[2], p[1]))
  } else if (parameter == "alpha") {
    # alpha [iter, topic-1, time, p_dim]
    p_names = expand.grid(p = 1:dims[4], t = 1:dims[3], z = 1:dims[2])
    names_vec = apply(p_names, 1, function(p) sprintf("alpha[z%d,t%d,p%d]", p[3], p[2], p[1]))
  } else {
    names_vec = paste0(parameter, "[", 1:total_params, "]")
  }

  dimnames(out) = list(NULL, "chain:1", names_vec)
  return(out)
}
