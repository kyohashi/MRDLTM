test_that("sample_u correctly samples utilities", {
  # --- setup ---
  active_data = data.frame(
    cust = c(1,1),
    item = c(1,1),
    time = c(1,2),
    y_cit = c(1,0)
  )

  state = init_state(
    active_data = active_data,
    n_item = 1,
    n_cust = 1,
    n_topic = 2,
    length_time = 2,
    n_var = 1,
    p_dim = 1
  )

  x_it <- array(c(1, 0.5), dim = c(1, 2, 2))

  # --- test ---
  sample_u(active_data, state, x_it)
  expect_true(state$u_cit[1] > 0) # y=1
  expect_true(state$u_cit[2] < 0) # y=0
})
