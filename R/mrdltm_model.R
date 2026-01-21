#' Define MR-DLTM Specification
#'
#' @description
#' Constructor for an MR-DLTM object.
#'
#' @param n_topic Number of latent topics (Z)
#' @param p_dim Dimension of the DLM state vector alpha_zt.
#' @param Dc Customer-level p-dimensional covariates such as demographics. If NULL, defaults to a Local Level model(p_dim = 1, Ft = 1).
#' @param Gt System matrix [p_dim x p_dim]. Defaults to a local level model.
#' @param ... Additional hyperparameters for priors.
#'
#' @return An object of class "mrdltm_model"
#' @export
mrdltm_model = function(n_topic = 3, p_dim = 1, Dc = NULL, Gt = NULL, ...){

  # Default Gt (Local level)
  if (is.null(Gt)) Gt = diag(1, p_dim)

  # Default Dc
  if (is.null(Dc)) {
    # Simple Local Level: Dc will be constructed later based on unique customers
    use_custom_D = FALSE
  } else {
    if (ncol(Dc) != p_dim) stop("ncol(Dc) must match p_dim")
    use_custom_D = TRUE
  }

  model = list(
    n_topic = n_topic,
    p_dim = p_dim,
    Gt = Gt,
    Dc = Dc,
    use_custom_D = use_custom_D,
    priors = list(...)
  )
  class(model) = "mrdltm_model"
  return(model)
}
