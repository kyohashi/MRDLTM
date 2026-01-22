test_that("sample_u correctly samples utilities", {
  # --- setup ---
  active_data = data.frame(
    cust = c(1,1),
    item = c(1,1),
    time = c(1,2),
    y_cit = c(1,0)
  )

  n_topic = 2
  n_item = 1
  n_var = 1

  state = init_state(
    active_data = active_data,
    n_item = n_item,
    n_cust = 1,
    n_topic = n_topic,
    length_time = 2,
    n_var = n_var,
    p_dim = 1
  )

  x_it <- array(c(1, 0.5), dim = c(1, 2, 2))

  # --- test ---
  sample_u(active_data, state, x_it, n_item, n_topic, n_var)
  expect_true(state$u_cit[1] > 0) # y=1
  expect_true(state$u_cit[2] < 0) # y=0
})
