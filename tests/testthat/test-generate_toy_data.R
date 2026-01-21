test_that("generate_toy_data produces correct dimensions and structure", {
  n_cust = 5
  n_item = 10
  n_topic = 3
  length_time = 12
  n_var = 2
  p_dim = 2

  toy = generate_toy_data(
    n_cust = n_cust,
    n_item = n_item,
    n_topic = n_topic,
    length_time = length_time,
    n_var = n_var,
    p_dim = p_dim
  )

  # 1. Check Overall Structure
  expect_named(toy, c("observations", "true_params"))
  expect_named(toy$observations, c("data", "x_it", "Dc"))

  # 2. Check Observation Dimensions
  # data: Total rows should be C * I * T
  expect_equal(nrow(toy$observations$data), n_cust * n_item * length_time)

  # x_it: [item, time, n_var]
  expect_equal(dim(toy$observations$x_it), c(n_item, length_time, n_var))

  # Dc: [n_cust, p_dim]
  expect_equal(dim(toy$observations$Dc), c(n_cust, p_dim))

  # 3. Check Identifiability Constraints in True Params
  # alpha_zt should be for Z-1 topics (for occupancy model)
  expect_equal(dim(toy$true_params$alpha_zt)[1], n_topic - 1)
  expect_equal(dim(toy$true_params$alpha_zt)[3], p_dim)

  # beta_zi should be for all Z topics (for response model)
  expect_equal(dim(toy$true_params$beta_zi)[1], n_topic)
  expect_equal(dim(toy$true_params$beta_zi)[3], n_var)
})

test_that("generate_toy_data respects intercept defaults", {
  # When n_var = 1 and p_dim = 1, x_it and Dc should be all 1s
  toy = generate_toy_data(n_var = 1, p_dim = 1)

  expect_true(all(toy$observations$x_it == 1))
  expect_true(all(toy$observations$Dc == 1))
})

test_that("generate_toy_data works with the model constructor", {
  # This tests the "Data-Bound" integration
  toy = generate_toy_data(n_topic = 4, p_dim = 2)

  # Passing toy$observations directly into mrdltm_model
  model = mrdltm_model(observations = toy$observations, n_topic = 4)

  expect_s3_class(model, "mrdltm_model")
  expect_equal(model$p_dim, 2)
  expect_true(model$use_custom_D)
})
