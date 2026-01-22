test_that("sample_eta runs without error and updates values", {
  # --- Setup ---
  n_cust = 3
  n_topic = 3
  length_time = 5
  n_item = 2
  p_dim = 2

  active_data = data.frame(
    cust = sample(1:n_cust, 10, replace = TRUE),
    item = sample(1:n_item, 10, replace = TRUE),
    time = sample(1:length_time, 10, replace = TRUE),
    y_cit = rbinom(10, 1, 0.5)
  )

  state = init_state(active_data, n_item, n_cust, n_topic, length_time, 1, p_dim)
  Dc = matrix(rnorm(n_cust * p_dim), n_cust, p_dim)

  # Initial values
  initial_eta = state$eta_zct + 0

  # --- Execution ---
  expect_error(
    sample_eta(active_data, state, Dc, n_cust, n_topic, length_time, p_dim),
    NA
  )

  # --- Verification ---
  # eta should be updated from its initial zeros
  expect_false(identical(state$eta_zct, initial_eta))
  expect_equal(dim(state$eta_zct), c(n_topic - 1, n_cust, length_time))
  expect_equal(dim(state$omega_zct), c(n_topic - 1, n_cust, length_time))
})
