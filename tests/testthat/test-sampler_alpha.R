test_that("sample_alpha runs without error and respects priors", {
  # --- Setup ---
  n_topic = 2
  length_time = 3
  p_dim = 1
  n_cust = 5

  state = new.env()
  state$alpha_zt = array(0, dim = c(n_topic - 1, length_time, p_dim))
  state$eta_zct = array(rnorm(1 * n_cust * length_time), dim = c(1, n_cust, length_time))
  state$g2_z = rep(0.1, 1)
  state$a2_z = rep(0.1, 1)

  Dc = matrix(1, nrow = n_cust, ncol = p_dim)

  # Custom prior
  custom_mz0 = 100.0
  priors = list(mz0 = custom_mz0, Sz0 = diag(0.01, p_dim))

  # --- Execution ---
  expect_error(sample_alpha(state, Dc, n_topic, length_time, p_dim, priors), NA)

  # --- Verification ---
  # Since Sz0 is very small and mz0 is 100, alpha_t should be pulled towards 100
  expect_gt(mean(state$alpha_zt[1,,]), 0)
  expect_equal(dim(state$alpha_zt), c(1, length_time, p_dim))
})
