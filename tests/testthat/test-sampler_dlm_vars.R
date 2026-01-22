test_that("sample_dlm_vars updates variances without error", {
  # --- Setup ---
  n_topic = 2
  length_time = 5
  p_dim = 2
  n_cust = 10

  state = new.env()
  state$alpha_zt = array(rnorm(1 * length_time * p_dim), dim = c(1, length_time, p_dim))
  state$eta_zct = array(rnorm(1 * n_cust * length_time), dim = c(1, n_cust, length_time))
  state$a2_z = rep(1.0, 1)
  state$b2_z = rep(1.0, 1)

  Dc = matrix(rnorm(n_cust * p_dim), n_cust, p_dim)
  priors = list()

  # --- Execution ---
  expect_error(sample_dlm_vars(state, Dc, n_topic, length_time, n_cust, p_dim, priors), NA)

  # --- Verification ---
  expect_length(state$a2_z, 1)
  expect_length(state$b2_z, 1)
  # Check if values are updated from 1.0
  expect_true(state$a2_z[1] != 1.0)
  expect_true(state$b2_z[1] != 1.0)
})
