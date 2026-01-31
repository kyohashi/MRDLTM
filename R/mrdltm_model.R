#' Define MR-DLTM Specification
#'
#' @description
#' Constructor for an MR-DLTM object.
#'
#' @param observations A list containing:
#'   \itemize{
#'     \item \code{data}: data.frame with (cust, item, time, y_cit)
#'     \item \code{x_it}: 3D array of marketing covariates (item, time, n_var)
#'     \item \code{Dc}: matrix of customer covariates (n_cust, p_dim). If NULL, defaults to local level.
#'   }
#' @param n_topic Number of latent topics (Z).
#' @param Gt System matrix (p_dim x p_dim). Defaults to a local level model.
#' @param ... Additional hyperparameters for priors.
#'
#' @return An object of class "mrdltm_model".
#' @export
mrdltm_model = function(observations, n_topic = 3, Gt = NULL, ...){

  # Default Dc
  if (is.null(observations$Dc)) {
    n_cust = length(unique(observations$data$cust))
    observations$Dc = matrix(1, nrow = n_cust, ncol = 1) # Default: Local Level Model
    p_dim = 1
    use_custom_D = FALSE
  } else {
    p_dim = ncol(observations$Dc)
    use_custom_D = TRUE
  }

  # Default Gt (Local level)
  if (is.null(Gt)) Gt = diag(1, p_dim)

  # Validation
  if (nrow(Gt) != p_dim) stop("Gt dimension must match p_dim from Dc.")

  model = list(
    observations = observations,
    n_topic = n_topic,
    p_dim = p_dim,
    Gt = Gt,
    use_custom_D = use_custom_D,
    priors = list(...)
  )
  class(model) = "mrdltm_model"
  return(model)
}
