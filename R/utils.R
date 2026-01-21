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
  state <- new.env(parent = emptyenv())

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
  state$omega_zct <- array(1, dim = c(n_z_dlm, n_cust, length_time))
  state$kappa_zct <- array(0, dim = c(n_z_dlm, n_cust, length_time))

  return(state)
}

#' Filter for Active Observations (Ic and Tc)
#'
#' @description
#' Filters the data to include only observations where:
#' 1. The item i belongs to Ic (items purchased by customer c at least once).
#' 2. The time t belongs to Tc (time points where customer c made at least one purchase).
#'
#' @param data A data frame containing 'cust', 'item', 'time', and 'y_cit'.
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
