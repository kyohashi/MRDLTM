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
reorder_mrdltm = function(res, burnin = 0) {

  n_iter  = dim(res$beta_zi)[1]
  n_topic = dim(res$beta_zi)[2]
  n_item  = dim(res$beta_zi)[3]
  n_var   = dim(res$beta_zi)[4]
  n_obs   = dim(res$z_cit)[2]

  # Target iterations after burn-in
  idx = (burnin + 1):n_iter
  m_post = length(idx)

  if (m_post <= 0) stop("Burn-in period exceeds total iterations.")

  # Store original dimensions to prevent dimension dropping errors
  d_eta_orig   = dim(res$eta_zct)   # [Iter, Z-1, Cust, Time]
  d_alpha_orig = dim(res$alpha_zt)  # [Iter, Z-1, Time, P]

  # --- 2. Prepare Inputs for label.switching ---
  # Reshape beta to [m_post x K x Parameters]
  mcmc_input = array(0, dim = c(m_post, n_topic, n_item * n_var))
  for (i in seq_len(m_post)) {
    m = idx[i]
    for (k in seq_len(n_topic)) {
      mcmc_input[i, k, ] = as.vector(res$beta_zi[m, k, , ])
    }
  }

  # Classification matrix
  z_input = res$z_cit[idx, , drop = FALSE]

  # Use maximum log-likelihood iteration as the pivot
  pivot_m = which.max(res$log_lik[idx])
  prapivot_input = mcmc_input[pivot_m, , ]

  # --- 3. Run Relabelling Algorithm (PRA) ---
  # Note: PRA is chosen for its stability in coordinate-based models
  ls_res = label.switching::label.switching(
    method   = "PRA",
    mcmc     = mcmc_input,
    prapivot = prapivot_input,
    z        = z_input,
    K        = n_topic
  )

  # perms[i, old_topic] = new_topic
  perms = as.matrix(ls_res$permutations$PRA)

  # --- 4. Parameter Reconstruction ---
  res_new = res

  for (i in seq_len(m_post)) {
    m = idx[i]
    p_vec = as.numeric(perms[i, ]) # Current permutation
    inv_p = order(p_vec)           # Map from new slot back to old topic

    # Identify the original topic that now serves as the Baseline (Topic Z)
    k_orig_base = inv_p[n_topic]

    # A. Reorder Beta and Z (Simple Permutation)
    res_new$beta_zi[m, , , ] = res$beta_zi[m, inv_p, , ]
    res_new$z_cit[m, ] = p_vec[res$z_cit[m, ]]

    # B. Adjust Alpha and Eta (Permutation + Baseline Shift)

    # Extract current iteration slice and manually fix dimensions to prevent dropping
    eta_slice = res$eta_zct[m, , , , drop = FALSE]
    dim(eta_slice) = d_eta_orig[-1] # [Z-1, Cust, Time]

    alpha_slice = res$alpha_zt[m, , , , drop = FALSE]
    dim(alpha_slice) = d_alpha_orig[-1] # [Z-1, Time, P]

    # Create full-topic arrays including the zero baseline
    full_eta   = array(0, dim = c(n_topic, d_eta_orig[3], d_eta_orig[4]))
    full_alpha = array(0, dim = c(n_topic, d_alpha_orig[3], d_alpha_orig[4]))

    full_eta[1:(n_topic - 1), , ]   = eta_slice
    full_alpha[1:(n_topic - 1), , ] = alpha_slice

    # Calculate the value of the topic that is the NEW baseline
    # Shifting relative to this ensures Topic Z remains 0
    shift_eta   = full_eta[k_orig_base, , , drop = FALSE]
    shift_alpha = full_alpha[k_orig_base, , , drop = FALSE]

    for (new_k in seq_len(n_topic - 1)) {
      orig_k = inv_p[new_k]
      res_new$eta_zct[m, new_k, , ]  = full_eta[orig_k, , ] - shift_eta[1, , ]
      res_new$alpha_zt[m, new_k, , ] = full_alpha[orig_k, , ] - shift_alpha[1, , ]
    }

    # C. Reorder Hierarchical Variances (a2_z and b2_z)
    # Since baseline topic has no variance in res, use mean as filler for reordering
    reorder_variance = function(vec, inv_p, n_topic) {
      v_full = c(vec, mean(vec))
      return(v_full[inv_p[1:(n_topic - 1)]])
    }

    res_new$a2_z[m, ] = reorder_variance(res$a2_z[m, ], inv_p, n_topic)
    if (!is.null(res$b2_z)) {
      res_new$b2_z[m, ] = reorder_variance(res$b2_z[m, ], inv_p, n_topic)
    }
  }

  return(res_new)
}
