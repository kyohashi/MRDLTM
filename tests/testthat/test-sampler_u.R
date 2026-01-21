test_that("sample_u correctly samples utilities", {
  # --- setup ---
  active_data = data.frame(
    cust = c(1,1),
    item = c(1,1),
    time = c(1,2),
    y_cit = c(1,0)
  )

  state = list(
    z_cit = c(1,1),
    beta_zi = array(0.5, dim = c(1,1,1)) # Topic 1, Item 1, Intercept Only
  )

  x_it = array(1, dim = c(1, 2, 1))

  # --- sample ---
  u_samples = sample_u(active_data, state, x_it)

  # --- test ---
  expect_length(u_samples, 2)
  expect_true(u_samples[1] > 0) # y=1
  expect_true(u_samples[2] <= 0) # y=0
})
