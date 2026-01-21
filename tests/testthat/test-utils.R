test_that("filter_active_data correctly identifies Ic and Tc", {
  # Setup toy data
  # Customer 1:
  #   - buys item 1 at t=1
  #   - does NOT buy item 1 at t=2
  #   - never buys item 2
  # Ic = {1}, Tc = {1}
  # Expected: Only (cust=1, item=1, t=1) should remain.

  df = data.frame(
    cust  = c(1, 1, 1, 1),
    item  = c(1, 1, 2, 2),
    time  = c(1, 2, 1, 2),
    y_cit = c(1, 0, 0, 0)
  )

  res = filter_active_data(df)

  expect_equal(nrow(res), 1)
  expect_equal(res$cust[1], 1)
  expect_equal(res$item[1], 1)
  expect_equal(res$time[1], 1)
})
