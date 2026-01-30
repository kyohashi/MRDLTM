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
#' Aligns topic labels across MCMC iterations using `label.switching` package.
#' Automatically selects the algorithm based on the availability of 'z_cit':
#' \itemize{
#'   \item If \code{store_z = TRUE}: Uses the **ECR algorithm**. Efficient for large Z but requires memory for Z.
#'   \item If \code{store_z = FALSE}: Uses the **PRA algorithm**. Standard method using parameters, slower for large Z.
#' }
#'
#' @param res A list containing MCMC samples from mrdltm_mcmc.
#' @param burnin Integer. Number of ORIGINAL iterations to discard.
#'
#' @return A list of MCMC samples with corrected labels.
#'
#' @importFrom label.switching label.switching
#' @export
reorder_mrdltm = function(res, burnin = 0) {

  # --- 1. Basic Dimensions with Thinning Awareness ---
  thin_rate = if (!is.null(res$thin)) res$thin else 1
  n_total_samples = dim(res$beta_zi)[1]
  n_topic = dim(res$beta_zi)[2]

  # Determine indices to keep based on original burnin count
  n_skip  = floor(burnin / thin_rate)
  idx     = (n_skip + 1):n_total_samples
  m_post  = length(idx)

  if (m_post <= 0) stop("Burn-in exceeds total iterations.")

  # Common: Find Pivot Index (Iteration with max log-likelihood)
  # Note: log_lik is usually full length, so we subset it to find the best in the post-burnin set
  log_lik_post = res$log_lik[idx]
  pivot_local_idx = which.max(log_lik_post) # Index relative to idx
  pivot_global_idx = idx[pivot_local_idx]   # Index relative to total samples

  perms = NULL

  # --- 2. Branching Logic based on store_z ---
  has_z = !is.null(res$z_cit)

  if (has_z) {
    # ==========================================
    # PATH A: ECR Algorithm (store_z = TRUE)
    # ==========================================
    cat("z_cit found. Running ECR algorithm (Method: ECR)...\n")

    # Extract Z for post-burnin only
    # res$z_cit is [Iter x Obs]
    z_subset = res$z_cit[idx, , drop = FALSE]

    # Pivot Z vector
    zpivot = z_subset[pivot_local_idx, ]

    # Run ECR
    # ECR uses the latent allocations to find the best permutation
    ls_res = label.switching::label.switching(
      method = "ECR",
      z      = z_subset,
      zpivot = zpivot,
      K      = n_topic
    )

    perms = as.matrix(ls_res$permutations$ECR)

    # Clean up memory
    rm(z_subset, ls_res); gc()

  } else {
    # ==========================================
    # PATH B: PRA Algorithm (store_z = FALSE)
    # ==========================================
    cat("z_cit NOT found. Running PRA algorithm (Method: PRA)...\n")

    # Prepare MCMC input for PRA (using Beta)
    # PRA requires: [MCMC Iterations x K x Parameters]
    n_item = dim(res$beta_zi)[3]
    n_var  = dim(res$beta_zi)[4]

    # Flatten Item x Var dimensions
    mcmc_input = array(0, dim = c(m_post, n_topic, n_item * n_var))
    for (k in seq_len(n_topic)) {
      mcmc_input[, k, ] = matrix(res$beta_zi[idx, k, , ], nrow = m_post)
    }

    # Dummy Z (required by label.switching function signature even for PRA,
    # though PRA relies on 'mcmc' and 'prapivot')
    # We create a minimal dummy that satisfies the check (must contain 1:K)
    z_dummy = matrix(rep(seq_len(n_topic), m_post), nrow = m_post, byrow = TRUE)

    # Pivot Parameter Matrix
    prapivot = mcmc_input[pivot_local_idx, , ]

    # Run PRA
    ls_res = label.switching::label.switching(
      method   = "PRA",
      mcmc     = mcmc_input,
      prapivot = prapivot,
      z        = z_dummy,
      K        = n_topic
    )

    perms = as.matrix(ls_res$permutations$PRA)

    # Clean up memory
    rm(mcmc_input, z_dummy, ls_res); gc()
  }

  # --- 3. Apply Corrections (Internal Helper Functions) ---

  # Helper: Shift logic for Location Parameters (alpha, eta) where Baseline = 0
  apply_rel_shift_update = function(arr, perms, n_topic, idx) {
    d_orig = dim(arr)
    out_dims = d_orig; out_dims[1] = length(idx)

    if (d_orig[2] != n_topic - 1) return(arr[idx, , , , drop=FALSE])

    result = array(0, dim = out_dims)
    for (i in seq_len(length(idx))) {
      m = idx[i]
      p_row = perms[i, ]; inv_p = order(p_row)
      k_orig_base = inv_p[n_topic] # Original topic assigned to New Baseline

      slice_dims = d_orig[-1]; slice_dims[1] = n_topic
      full_slice = array(0, dim = slice_dims)

      if (length(d_orig) == 4) {
        full_slice[1:(n_topic - 1), , ] = arr[m, , , ]
        shift_val = full_slice[k_orig_base, , , drop = FALSE]
        for (new_k in seq_len(n_topic - 1)) {
          orig_k = inv_p[new_k]
          result[i, new_k, , ] = full_slice[orig_k, , ] - shift_val[1, , ]
        }
      } else {
        full_slice[1:(n_topic - 1), ] = arr[m, , ]
        shift_val = full_slice[k_orig_base, , drop = FALSE]
        for (new_k in seq_len(n_topic - 1)) {
          orig_k = inv_p[new_k]
          result[i, new_k, ] = full_slice[orig_k, ] - shift_val[1, ]
        }
      }
    }
    return(result)
  }

  # Helper: Reorder logic for K-1 Parameters (mu, phi, sigma, a2) without shift
  apply_k_minus_1_reorder = function(arr, perms, n_topic, idx, impute_fn = mean) {
    d_orig = dim(arr); out_dims = d_orig; out_dims[1] = length(idx)
    result = array(0, dim = out_dims)

    for (i in seq_len(length(idx))) {
      m = idx[i]; inv_p = order(perms[i, ])
      vals = arr[m, , drop=FALSE]
      if (length(d_orig) == 2) {
        full_vec = c(vals, impute_fn(vals))
        result[i, ] = full_vec[inv_p[1:(n_topic - 1)]]
      } else {
        result[i, , ] = vals
      }
    }
    return(result)
  }

  # Helper: Standard Reordering for K-dim Parameters
  apply_standard_reorder = function(arr, perms, idx) {
    d_orig = dim(arr); out_dims = d_orig; out_dims[1] = length(idx)
    result = array(0, dim = out_dims)
    for (i in seq_len(length(idx))) {
      m = idx[i]; inv_p = order(perms[i, ])
      if (length(d_orig) == 3) result[i, , ] = arr[m, inv_p, ]
      else if (length(d_orig) == 4) result[i, , , ] = arr[m, inv_p, , ]
      else result[i, ] = arr[m, inv_p]
    }
    return(result)
  }

  # --- 4. MAIN LOOP Over All Elements in List ---
  cat("Applying corrections to all parameters...\n")
  res_out = list()

  for (name in names(res)) {
    obj = res[[name]]
    if (is.null(obj)) next

    # 1. Handle Z_CIT (Values need mapping if it exists)
    if (name == "z_cit") {
      # Only process if it's a matrix (i.e., was stored)
      if (is.matrix(obj)) {
        n_obs = ncol(obj)
        z_out = matrix(0L, nrow = m_post, ncol = n_obs)
        for (i in seq_len(m_post)) {
          m = idx[i]; p_vec = perms[i, ]
          z_out[i, ] = p_vec[obj[m, ]]
        }
        res_out[[name]] = z_out
      }

      # 2. Handle Location Parameters (Alpha, Eta, Mu) -> Shift Logic
    } else if (name %in% c("alpha_zt", "eta_zct", "mu_zt")) {
      res_out[[name]] = apply_rel_shift_update(obj, perms, n_topic, idx)

      # 3. Handle Standard K-dim Parameters (beta_zi etc.)
    } else if (is.array(obj) && length(dim(obj)) >= 2 && dim(obj)[2] == n_topic) {
      res_out[[name]] = apply_standard_reorder(obj, perms, idx)

      # 4. Handle Other K-1 Parameters (Variance/Phi etc.)
    } else if (name %in% c("a2_z", "phi_z", "sigma_z", "b2_z") &&
               is.array(obj) && dim(obj)[2] == n_topic - 1) {
      res_out[[name]] = apply_k_minus_1_reorder(obj, perms, n_topic, idx)

      # 5. Default: Remove burn-in and retain metadata
    } else {
      if (name == "thin") {
        res_out[[name]] = obj
      } else if (is.vector(obj) && length(obj) == n_total_samples) {
        res_out[[name]] = obj[idx]
      } else if (is.array(obj) && dim(obj)[1] == n_total_samples) {
        args = rep(list(bquote()), length(dim(obj))); args[[1]] = idx
        res_out[[name]] = do.call("[", c(list(obj), args, drop=FALSE))
      } else {
        res_out[[name]] = obj
      }
    }
  }

  class(res_out) = class(res)
  cat("Label switching correction complete.\n")
  return(res_out)
}
