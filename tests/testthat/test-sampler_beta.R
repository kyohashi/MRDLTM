test_that("sample_beta updates coefficients based on latent utility", {
  # --- Setup ---
  # Simple case: 1 customer, 1 item, 2 time points, 1 marketing var
  active_data = data.frame(
    cust = c(1, 1),
    item = c(1, 1),
    time = c(1, 2),
    y_cit = c(1, 1)
  )

  n_topic = 2
  n_item = 1
  n_var = 1
  length_time = 2

  state = init_state(
    active_data = active_data,
    n_item = n_item,
    n_cust = 1,
    n_topic = n_topic,
    length_time = length_time,
    n_var = n_var,
    p_dim = 1
  )

  # Assign all observations to Topic 1
  state$z_cit = c(1, 1)

  # Set positive latent utilities to pull beta upwards
  state$u_cit = c(5.0, 5.0)

  # Constant covariate (e.g., intercept only)
  x_it = array(1, dim = c(1, 2, 1))

  # Store initial values
  initial_beta = state$beta_zi + 0

  # --- Execution ---
  sample_beta(active_data, state, x_it, n_item, n_topic, n_var, length_time)

  # --- Verification ---
  # beta_zi should be updated
  expect_false(identical(state$beta_zi, initial_beta))

  # Topic 1's beta should be positive due to high utility
  expect_gt(state$beta_zi[1, 1, 1], 0)

  # Topic 2 has no data, so it should be sampled from prior N(0, 1)
  # Just checking it exists and is numeric
  expect_true(is.numeric(state$beta_zi[2, 1, 1]))
})
