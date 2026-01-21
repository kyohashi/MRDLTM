test_that("mrdltm_model initializes with correct defaults", {
  model = mrdltm_model(n_topic = 5, p_dim = 2)

  expect_s3_class(model, "mrdltm_model")
  expect_equal(model$n_topic, 5)
  expect_equal(model$p_dim, 2)
  expect_equal(model$Gt, diag(1,2))
  expect_false(model$use_custom_D)
})

test_that("mrdltm_model handles custom Dc correctly", {
  p = 3
  n_cust = 10
  dummy_Dc = matrix(rnorm(n_cust * p), nrow = n_cust, ncol = p)

  model = mrdltm_model(p_dim = p, Dc = dummy_Dc)

  expect_true(model$use_custom_D)
  expect_equal(model$Dc, dummy_Dc)
  expect_equal(ncol(model$Dc), p)
})

test_that("mrdltm_model throws error on dimension mismatch", {
  p = 2
  invalid_Dc = matrix(0, nrow = 5, ncol = 3)

  expect_error(
    mrdltm_model(p_dim = p, Dc = invalid_Dc)
  )
})

test_that("mrdltm_model captures additional hyperparams in ...", {
  model = mrdltm_model(n_topic = 3, a0 = 0.01, b0 = 0.01)

  expect_equal(model$priors$a0, 0.01)
  expect_equal(model$priors$b0, 0.01)
})
