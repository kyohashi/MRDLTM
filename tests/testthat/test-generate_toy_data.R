test_that("generate_toy_data returns correct dimensions and types", {
  n_cust = 10
  n_item = 50
  n_topic = 3
  length_time = 30
  n_var = 2

  toy = generate_toy_data(
    n_cust = n_cust,
    n_item = n_item,
    n_topic = n_topic,
    length_time = length_time,
    n_var = n_var
  )

  # 1. Check Output Structure
  expect_type(toy, "list")
  expect_named(toy, c("observations", "true_params"))

  # 2. Check Observation Dimensions
  # expand.grid returns n_cust * n_item * length_time rows
  expect_equal(nrow(toy$observations$data), n_cust * n_item * length_time)
  expect_equal(dim(toy$observations$x_it), c(n_item, length_time, n_var))

  # 3. Check Statistical Invariants
  # Topic proportions must sum to 1 across topics for each time point
  expect_equal(colSums(toy$true_params$theta_zt), rep(1, length_time))

  # Purchase data should be binary (0 or 1)
  expect_true(all(toy$observations$data$y_cit %in% c(0, 1)))

  # 4. Check Hierarchical Beta dimensions
  expect_equal(dim(toy$true_params$beta_zi), c(n_topic, n_item, n_var))

})
