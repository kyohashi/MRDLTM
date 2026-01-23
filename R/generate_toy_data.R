#' Generate Toy Data for MR-DLTM
#'
#' @description
#' Generates synthetic data based on the Market Response Dynamic Linear Topic Model(MR-DLTM).
#'
#' @param n_cust Number of customers (C)
#' @param n_item Number of items (I)
#' @param n_topic Number of latent topics (Z)
#' @param length_time Length of time points (T)
#' @param n_var Number of marketing covariates including intercept (M)
#' @param p_dim Dimension of customer covariates including intercept (P)
#'
#' @return A list containing:
#' \itemize{
#'   \item observations: List of observed data (data, x_it, Dc)
#'   \item true_params: List of ground truth parameters
#' }
#' @importFrom stats rnorm
#' @export
generate_toy_data = function(n_cust = 10, n_item = 50, n_topic = 3, length_time = 30, n_var = 2, p_dim = 1) {
  # --- 1. Generate Marketing Covariates (x_it) ---
  # x_it: [item, time, n_var]
  x_it = array(rnorm(n_item*length_time*n_var), dim = c(n_item, length_time, n_var))
  # if (n_var > 1) {
  #   x_it[, , 2:n_var] = rnorm(n_item * length_time * (n_var - 1))
  # }

  # --- 2. Generate Customer Covariates (Dc) ---
  # Dc: [n_cust, p_dim]
  Dc = matrix(1, nrow = n_cust, ncol = p_dim)
  if (p_dim > 1) {
    Dc[, 2:p_dim] = rnorm(n_cust * (p_dim - 1))
  }

  # --- 3. Generate Response Coefficients (beta_zi) ---
  # beta_zi ~ N(mu_i, V_i)
  mu_i = matrix(0, nrow = n_item, ncol = n_var)
  V_i = 1.0
  beta_zi = array(0, dim = c(n_topic, n_item, n_var))
  for (z in 1:n_topic) {
    for (i in 1:n_item) {
      beta_zi[z, i, ] = mu_i[i, ] + rnorm(n_var, sd = sqrt(V_i))
    }
  }

  # --- 4. Generate Dynamic Topic Occupancy (DLM) ---
  # We generate Z-1 processes for identifiability (the Z-th topic is the baseline)
  n_z_dlm = n_topic - 1
  a2_z = rep(0.05, n_z_dlm) # Observation variance
  b2_z = rep(0.01, n_z_dlm) # System variance

  # alpha_zt: [Z-1, T, p_dim]
  alpha_zt = array(0, dim = c(n_z_dlm, length_time, p_dim))
  for (z in 1:n_z_dlm) {
    # Initial state alpha_z,t=1
    alpha_zt[z, 1, ] = rnorm(p_dim, sd = 0.5)
    for (t in 2:length_time) {
      # Random walk: alpha_t = alpha_{t-1} + N(0, b2_z)
      alpha_zt[z, t, ] = alpha_zt[z, t - 1, ] + rnorm(p_dim, sd = sqrt(b2_z[z]))
    }
  }

  # --- 5. Generate Latent Occupancy (eta) and Probabilities (theta) ---
  # eta_zct = Dc %*% alpha_zt + epsilon
  theta_czt = array(0, dim = c(n_cust, length_time, n_topic))

  for (t in 1:length_time) {
    # Store eta for Z-1 topics
    eta_tmp = matrix(0, nrow = n_cust, ncol = n_topic) # Last column stays 0 for baseline
    for (z in 1:n_z_dlm) {
      # Linear predictor: Dc [C x P] %*% alpha_zt [P x 1]
      mu_eta = Dc %*% alpha_zt[z, t, ]
      eta_tmp[, z] = mu_eta + rnorm(n_cust, sd = sqrt(a2_z[z]))
    }

    # Softmax to get theta_czt
    # exp(eta) / sum(exp(eta))
    exp_eta = exp(eta_tmp)
    theta_czt[, t, ] = exp_eta / rowSums(exp_eta)
  }

  # --- 6. Generate Purchase Data (y_cit) ---
  total_obs = n_cust * n_item * length_time
  indices = expand.grid(cust = 1:n_cust, item = 1:n_item, time = 1:length_time)

  y_cit = integer(total_obs)
  z_cit = integer(total_obs)
  u_cit = numeric(total_obs)

  for (n in 1:total_obs) {
    c_idx = indices$cust[n]
    i_idx = indices$item[n]
    t_idx = indices$time[n]

    # Sample latent topic z_cit based on customer-time specific theta
    z_cit[n] = sample(1:n_topic, size = 1, prob = theta_czt[c_idx, t_idx, ])

    # Calculate utility: u = x'beta + epsilon
    x_vec = x_it[i_idx, t_idx, ]
    beta_vec = beta_zi[z_cit[n], i_idx, ]
    u_cit[n] = sum(beta_vec * x_vec) + rnorm(1, mean = 0, sd = 1)

    # Final binary purchase observation
    y_cit[n] = as.integer(u_cit[n] > 0)
  }

  observed_df = cbind(indices, y_cit = y_cit)

  return(
    list(
      observations = list(
        data = observed_df,
        x_it = x_it,
        Dc = Dc
      ),
      true_params = list(
        mu_i = mu_i,
        V_i = V_i,
        beta_zi = beta_zi,
        a2_z = a2_z,
        b2_z = b2_z,
        alpha_zt = alpha_zt,
        theta_czt = theta_czt,
        z_cit = z_cit,
        u_cit = u_cit
      )
    )
  )
}
