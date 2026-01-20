#' Generate Toy Data for MR-DLTM
#'
#' @description
#' Generates synthetic data based on the Market Response Dynamic Linear Topic Model(MR-DLTM).
#'
#' @param n_cust Number of customers (C)
#' @param n_item Number of items (I)
#' @param n_topic Number of latent topics (Z)
#' @param length_time Length of time points (T)
#' @param n_var Number of marketing covariates (M)
#'
#' @return A list containing:
#' \itemize{
#'   \item {observed}{List of observed data (y, x)}
#'   \item {true_params}{List of ground truth parameters}
#' }
#' @export
generate_toy_data = function(n_cust = 10, n_item = 50, n_topic = 3, length_time = 30, n_var = 2) {
  set.seed(42)

  # --- Generate Marketing Covariates (x_it) ---
  x_wo_intercept = array(rnorm(n_item * length_time * n_var), dim = c(n_item, length_time, n_var))
  ## add constant
  x_it = array(1, dim = c(n_item, length_time, n_var + 1))
  x_it[, , 2:(n_var + 1)] = x_wo_intercept # Fill the rest
  dim_var = n_var + 1

  # --- Generate Response Coefficients (beta_zi) ---
  ## beta_zi ~ N(mu_i, V_i)
  mu_i = matrix(rnorm(n_item * dim_var, mean = 0, sd = 0.5), nrow = n_item, ncol = dim_var)
  V_i = 0.2 # Common variance across items
  beta_zi = array(0, dim = c(n_topic, n_item, dim_var))
  for (z in 1:n_topic){
    for (i in 1:n_item){
      beta_zi[z, i, ] = mu_i[i, ] + rnorm(dim_var, sd = sqrt(V_i))
    }
  }

  # --- Generate Dynamic Topic Proportions ---
  ## DLM variances
  a_z = 0.05
  b_z = 0.1

  ## System Model: alpha_zt = alpha_{z, t-1} + N(0, b_z)
  alpha_zt = matrix(0, nrow = n_topic, ncol = length_time)
  for (t in 2:length_time){
    alpha_zt[, t] = alpha_zt[, t-1] + rnorm(n_topic, sd = sqrt(b_z))
  }

  ## Observation Model: eta_zt = alpha_zt + N(0, a_z)
  eta_zt = matrix(0, nrow = n_topic, ncol = length_time)
  for (t in 1:length_time){
    eta_zt[, t] = alpha_zt[, t] + rnorm(n_topic, sd = sqrt(a_z))
  }

  ## theta_zt: dynamic topic proportions are same across customers for simplicity
  theta_zt = apply(eta_zt, 2, function(a) exp(a) / sum(exp(a)))

  # --- Generate Purchase Data ---
  ## define containers
  total_obs = n_cust * n_item * length_time
  y_cit = integer(total_obs)
  z_cit = integer(total_obs)
  u_cit = numeric(total_obs)

  indices = expand.grid(cust = 1:n_cust, item = 1:n_item, time = 1:length_time)
  for (n in 1:total_obs){
    c_idx = indices$cust[n]
    i_idx = indices$item[n]
    t_idx = indices$time[n]

    ## sample latent topic z_cit
    z_cit[n] = sample(1:n_topic, size = 1, prob = theta_zt[, t_idx])

    ## calculate utility
    x_vec = x_it[i_idx, t_idx, ]
    beta_vec = beta_zi[z_cit[n], i_idx, ]
    u_cit[n] = sum(beta_vec * x_vec) + rnorm(1, mean = 0, sd = 1)

    ## purchase observation
    y_cit[n] = as.integer(u_cit[n] > 0)
  }

  observed_df = cbind(indices, y_cit = y_cit)

  return(
    list(
      observations = list(
        data = observed_df,
        x_it = x_it
      ),
      true_params = list(
        mu_i = mu_i,
        V_i = V_i,
        beta_zi = beta_zi,
        a_z = a_z,
        b_z = b_z,
        alpha_zt = alpha_zt,
        eta_zt = eta_zt,
        theta_zt = theta_zt,
        z_cit = z_cit,
        u_cit = u_cit
      )
    )
  )
}
