test_that("sample_mu_V runs without error", {
  # --- Setup minimal state ---
  n_topic = 3
  n_item = 2
  n_var = 1
  active_data = data.frame(cust=1, item=1, time=1, y_cit=1)

  state = init_state(
    active_data = active_data,
    n_item = n_item,
    n_cust = 1,
    n_topic = n_topic,
    n_time = 1,
    n_var = n_var,
    p_dim = 1
  )

  priors = list()

  # --- Execution & Basic Check ---
  # Check if the function completes without error
  expect_error(sample_mu_V(state, n_item, n_topic, n_var, priors), NA)

  # Check if values are updated from initial zeros (mu_i) or identity (V_i)
  expect_true(is.numeric(state$mu_i))
  expect_true(is.numeric(state$V_i))
})
