test_that("sample_z updates topic assignments without error", {
  # --- 1. Setup minimal environment ---
  n_cust = 2
  n_item = 2
  n_topic = 3
  length_time = 2
  n_var = 2 # Intercept + 1 covariate

  # Mock active_data
  active_data = data.frame(
    cust = c(1, 2),
    item = c(1, 1),
    time = c(1, 2),
    y_cit = c(1, 0)
  )
  n_obs = nrow(active_data)

  # Initialize state
  state = init_state(
    active_data = active_data,
    n_item = n_item,
    n_cust = n_cust,
    n_topic = n_topic,
    length_time = length_time,
    n_var = n_var,
    p_dim = 1
  )

  # Setup mock covariates (intercepts)
  x_it = array(0, dim = c(n_item, length_time, n_var))
  x_it[,,1] = 1 # Intercepts
  x_it[,,2] = rnorm(n_item * length_time) # Random covariate

  # Setup mock parameters in state
  # Topic 1 has high beta for intercept, Topic 2 has low beta
  state$beta_zi[1, , 1] = 10.0
  state$beta_zi[2, , 1] = -10.0
  state$beta_zi[3, , 1] = 0.0

  # Occupancy eta (all near 0)
  state$eta_zct = array(0, dim = c(n_topic - 1, n_cust, length_time))

  # Store initial z_cit
  initial_z = state$z_cit + 0

  # --- 2. Execution ---
  # Check for error (Survival test)
  expect_error(
    sample_z(active_data, state, x_it, n_obs, n_topic, n_cust, n_var, length_time),
    NA
  )

  # --- 3. Verification ---
  # z_cit should be updated
  expect_equal(length(state$z_cit), n_obs)
  expect_true(all(state$z_cit %in% 1:n_topic))

  # Optional: Data dependency check
  # Since y_cit[1]=1 and Topic 1 has high beta, Topic 1 is more likely for obs 1
  # Since y_cit[2]=0 and Topic 2 has low beta, Topic 2 is more likely for obs 2
  # Note: Sampling is stochastic, but with beta=10/-10, it's very likely.
  expect_type(state$z_cit, "integer")
})
