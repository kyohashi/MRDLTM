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
as.array.mrdltm_mcmc = function(x, parameter = "log_lik", ...) {
  samples = x[[parameter]]
  if (is.null(samples)) stop(paste("Parameter", parameter, "not found in results."))

  dims = dim(samples)

  # Case: 1D Vector (like log_lik)
  if (is.null(dims)) {
    n_iter = length(samples)
    out = array(samples, dim = c(n_iter, 1, 1))
    dimnames(out) = list(NULL, "chain:1", parameter)
    return(out)
  }

  n_iter = dims[1]
  total_params = prod(dims[-1])

  # Flatten Topic/Item/Time dimensions into a single parameter dimension
  # Results in [Iterations x Chains(1) x Parameters]
  out = samples
  dim(out) = c(n_iter, 1, total_params)

  # --- Generate descriptive labels for each multi-dimensional index ---
  # expand.grid order matches R's column-major flattening (Topic -> Item/Time -> Var)
  if (parameter == "beta_zi") {
    # dims: [iter, topic, item, var]
    p_names = expand.grid(z = 1:dims[2], i = 1:dims[3], v = 1:dims[4])
    names_vec = sprintf("beta_zi[%d,%d,%d]", p_names$z, p_names$i, p_names$v)
  } else if (parameter == "alpha_zt") {
    # dims: [iter, topic, time, p]
    p_names = expand.grid(z = 1:dims[2], t = 1:dims[3], p = 1:dims[4])
    names_vec = sprintf("alpha_zt[%d,%d,%d]", p_names$z, p_names$t, p_names$p)
  } else if (parameter == "eta_zct") {
    # dims: [iter, topic, cust, time]
    p_names = expand.grid(z = 1:dims[2], c = 1:dims[3], t = 1:dims[4])
    names_vec = sprintf("eta_zct[%d,%d,%d]", p_names$z, p_names$c, p_names$t)
  } else if (parameter %in% c("a2_z", "b2_z")) {
    names_vec = sprintf("%s[%d]", parameter, 1:dims[2])
  } else {
    names_vec = paste0(parameter, "[", 1:total_params, "]")
  }

  dimnames(out) = list(
    iterations = NULL,
    chains = "chain:1",
    parameters = names_vec
  )

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
extract_samples = function(x, parameter = "log_lik", burnin = 0, thin = 1) {
  # Call the S3 method as.array to get the full array
  arr = as.array(x, parameter = parameter)

  iter_total = dim(arr)[1]

  # Selection logic
  indices = seq(from = burnin + 1, to = iter_total, by = thin)

  if (length(indices) == 0) {
    stop("No samples left after applying burn-in and thinning.")
  }

  # Subset along the first dimension (Iterations)
  return(arr[indices, , , drop = FALSE])
}

#' Post-hoc Label Switching Correction for MR-DLTM
#'
#' @description
#' Aligns topic labels across MCMC iterations using the Pivot Relabelling Algorithm (PRA).
#' This function also ensures the structural baseline constraint (Topic Z = 0)
#' is preserved by dynamically shifting the relative parameters (alpha and eta).
#'
#' @param res A list containing MCMC samples from mrdltm_mcmc.
#' @param burnin Integer. Number of initial iterations to discard.
#'
#' @return A list of MCMC samples with corrected labels and adjusted baselines.
#'
#' @importFrom label.switching label.switching
#' @export
reorder_mrdltm = function(res, burnin = 0, detection_subset_size = 20000) {

  # --- 1. Basic Dimensions ---
  n_iter  = dim(res$beta_zi)[1]
  n_topic = dim(res$beta_zi)[2]
  n_obs   = dim(res$z_cit)[2]
  idx     = (burnin + 1):n_iter
  m_post  = length(idx)

  if (m_post <= 0) stop("Burn-in exceeds total iterations.")

  # --- 2. Label Switching Detection (using a subset for memory safety) ---
  # we use all Beta parameters but only a subset of Z observations.

  cat("Preparing detection subset...\n")
  n_item = dim(res$beta_zi)[3]
  n_var  = dim(res$beta_zi)[4]
  mcmc_input = array(0, dim = c(m_post, n_topic, n_item * n_var))
  for (k in seq_len(n_topic)) {
    mcmc_input[, k, ] = matrix(res$beta_zi[idx, k, , ], nrow = m_post)
  }

  # Use a representative subset of observations to identify permutations
  obs_sample_idx = sample(seq_len(n_obs), size = min(n_obs, detection_subset_size))
  z_sample = res$z_cit[idx, obs_sample_idx, drop = FALSE]

  pivot_m = which.max(res$log_lik[idx])

  cat("Running PRA algorithm on subset...\n")
  ls_res = label.switching::label.switching(
    method   = "PRA",
    mcmc     = mcmc_input,
    prapivot = mcmc_input[pivot_m, , ],
    z        = z_sample,
    K        = n_topic
  )

  # Extract permutations [m_post x n_topic] mapping: old_label -> new_label
  perms = as.matrix(ls_res$permutations$PRA)

  # Clean up temporary detection objects
  rm(mcmc_input, z_sample, ls_res); gc()

  # --- 3. Apply Permutations to the FULL dataset ---

  # A. Align FULL z_cit matrix (The most memory-intensive part)
  cat("Aligning all observations (z_cit)...\n")
  for (i in seq_len(m_post)) {
    m = idx[i]
    p_vec = perms[i, ]
    # In-place update to avoid duplication
    res$z_cit[m, ] = p_vec[res$z_cit[m, ]]
  }
  gc()

  # B. Align and Shift Relative Parameters (alpha, eta)
  # This section handles the baseline shift to maintain Topic Z = 0

  # Helper for in-place relative parameter update
  apply_rel_update = function(arr, perms, n_topic, idx) {
    d_orig = dim(arr)
    for (i in seq_len(length(idx))) {
      m = idx[i]
      inv_p = order(perms[i, ])
      k_orig_base = inv_p[n_topic] # Which original topic is now the baseline

      # Working slice for current iteration
      slice_dims = d_orig[-1]; slice_dims[1] = n_topic
      full_slice = array(0, dim = slice_dims)
      full_slice[1:(n_topic - 1), , ] = arr[m, , , , drop = FALSE]

      shift_val = full_slice[k_orig_base, , , drop = FALSE]
      for (new_k in seq_len(n_topic - 1)) {
        orig_k = inv_p[new_k]
        arr[m, new_k, , ] = full_slice[orig_k, , ] - shift_val[1, , ]
      }
    }
    return(arr)
  }

  cat("Aligning and shifting alpha_zt...\n")
  res$alpha_zt = apply_rel_update(res$alpha_zt, perms, n_topic, idx)

  cat("Aligning and shifting eta_zct...\n")
  res$eta_zct = apply_rel_update(res$eta_zct, perms, n_topic, idx)
  gc()

  # C. Align Beta and Variances
  cat("Finalizing beta and variance alignment...\n")
  for (i in seq_len(m_post)) {
    m = idx[i]
    inv_p = order(perms[i, ])

    res$beta_zi[m, , , ] = res$beta_zi[m, inv_p, , ]

    a2_full = c(res$a2_z[m, ], mean(res$a2_z[m, ]))
    res$a2_z[m, ] = a2_full[inv_p[1:(n_topic - 1)]]
  }

  # --- 4. Subset to Burn-in and Return ---
  res$beta_zi  = res$beta_zi[idx, , , , drop = FALSE]
  res$z_cit    = res$z_cit[idx, , drop = FALSE]
  res$alpha_zt = res$alpha_zt[idx, , , , drop = FALSE]
  res$eta_zct  = res$eta_zct[idx, , , , drop = FALSE]
  res$a2_z     = res$a2_z[idx, , drop = FALSE]
  res$log_lik  = res$log_lik[idx]

  return(res)
}
