test_that("mrdltm_model initializes with correct defaults", {
  # Setup dummy observations without Dc
  n_cust = 10
  observations = list(
    data = data.frame(cust = 1:n_cust, y_cit = 0),
    x_it = array(0, dim = c(1, 1, 1)),
    Dc = NULL
  )

  # Initialize model
  model = mrdltm_model(observations = observations, n_topic = 5)

  # Assertions for default Local Level Model
  expect_s3_class(model, "mrdltm_model")
  expect_equal(model$n_topic, 5)
  expect_equal(model$p_dim, 1) # Default p_dim is 1
  expect_equal(model$Gt, matrix(1, 1, 1)) # Default Gt is diag(1)
  expect_false(model$use_custom_D)

  # Check if Dc was automatically created as a column of 1s
  expect_equal(nrow(model$observations$Dc), n_cust)
  expect_equal(ncol(model$observations$Dc), 1)
  expect_true(all(model$observations$Dc == 1))
})

test_that("mrdltm_model handles custom Dc correctly", {
  p_dim = 3
  n_cust = 10
  dummy_Dc = matrix(rnorm(n_cust * p_dim), nrow = n_cust, ncol = p_dim)

  # Setup observations with custom Dc
  observations = list(
    data = data.frame(cust = 1:n_cust),
    x_it = array(0, dim = c(1, 1, 1)),
    Dc = dummy_Dc
  )

  model = mrdltm_model(observations = observations, n_topic = 3)

  # Assertions for custom covariates
  expect_true(model$use_custom_D)
  expect_equal(model$p_dim, p_dim)
  expect_equal(model$observations$Dc, dummy_Dc)
  expect_equal(model$Gt, diag(1, p_dim))
})

test_that("mrdltm_model throws error on dimension mismatch between Dc and Gt", {
  p_dim = 2
  dummy_Dc = matrix(0, nrow = 5, ncol = p_dim)

  observations = list(
    data = data.frame(cust = 1:5),
    Dc = dummy_Dc
  )

  # Pass a 3x3 Gt while Dc implies p_dim = 2
  invalid_Gt = diag(1, 3)

  expect_error(
    mrdltm_model(observations = observations, Gt = invalid_Gt),
    "Gt dimension must match p_dim from Dc"
  )
})

test_that("mrdltm_model captures additional hyperparams in ...", {
  observations = list(
    data = data.frame(cust = 1),
    Dc = NULL
  )

  # Pass additional prior parameters
  model = mrdltm_model(observations = observations, n_topic = 3, a0 = 0.01, b0 = 0.01)

  expect_equal(model$priors$a0, 0.01)
  expect_equal(model$priors$b0, 0.01)
})
