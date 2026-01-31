test_that("sample_alpha runs without error and respects priors", {
  # --- Setup ---
  n_topic = 2
  n_time = 3
  p_dim = 1
  n_cust = 5

  state = new.env()
  state$alpha_zt = array(0, dim = c(n_topic - 1, n_time, p_dim))
  # Initialize with random values
  state$eta_zct = array(rnorm(1 * n_cust * n_time), dim = c(1, n_cust, n_time))
  state$b2_z = rep(0.1, 1)
  state$a2_z = rep(0.1, 1)

  Dc = matrix(1, nrow = n_cust, ncol = p_dim)

  # Custom prior: Strong prior to verify impact
  custom_mz0 = 100.0
  priors = list(mz0 = custom_mz0, Sz0 = diag(0.01, p_dim))

  # --- Create Dummy Active Data ---
  # For testing purposes, we assume all customers are active at all time points.
  # In a real scenario, this would be sparse.
  active_data = expand.grid(
    cust = 1:n_cust,
    time = 1:n_time
  )
  # Add dummy columns required if the wrapper checks them (though cpp only needs cust/time)
  active_data$y_cit = 1
  active_data$item = 1

  # --- Execution ---
  # Note: Ensure the R wrapper `sample_alpha` is updated to accept `active_data`
  # and pass `active_data$cust` and `active_data$time` to the C++ function.
  expect_error(sample_alpha(active_data, state, Dc, n_topic, n_time, p_dim, priors), NA)

  # --- Verification ---
  # Since Sz0 is very small and mz0 is 100, alpha_t should be pulled towards 100,
  # dominating the random noise from eta_zct.
  expect_gt(mean(state$alpha_zt[1,,]), 0)
  expect_equal(dim(state$alpha_zt), c(1, n_time, p_dim))
})
