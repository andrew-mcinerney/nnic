#' nn_fit_tracks
#'
#'
#' @param X input matrix
#' @param Y output vector
#' @param q number of hidden units
#' @param W_mat inital weights matrix
#' @param inf_crit information criterion (BIC only)
#' @param unif uniform distribution
#' @return nn_fit_tracks
#' @export
nn_fit_tracks <- function (X, Y, q, W_mat, inf_crit = 'BIC', unif = 3, ...){
  # Function with fits n_iter tracks of model and finds best

  df <- data.frame(X, Y)
  n_iter <- nrow(W_mat)
  k <- ncol(W_mat)
  n <- nrow(X)

  weight_matrix = matrix(rep(NA, n_iter*k), ncol = k)
  inf_crit_vec <- rep(NA, n_iter)
  converge <- rep(NA, n_iter)

  for(iter in 1:n_iter){
    nn_model =  nnet::nnet(Y~., data = df, size = q, trace = F, linout = T,
                           Wts = W_mat[iter, ], ...)
    weight_matrix[iter,] = nn_model$wts

    RSS = nn_model$value
    sigma2 = RSS/n
    log_likelihood = (-n/2)*log(2*pi*sigma2) - RSS/(2*sigma2)
    inf_crit_vec[iter] = ifelse(inf_crit == 'AIC', (2*(k+1) - 2*log_likelihood),
                                ifelse(inf_crit == 'BIC', (log(n)*(k+1) - 2*log_likelihood),
                                       ifelse(inf_crit == 'AICc', (2*(k+1)*(n/(n-(k+1)-1)) - 2*log_likelihood),NA)))
    converge[iter] = nn_model$convergence
  }
  W_opt = weight_matrix[which.min(inf_crit_vec), ]

  return(list(
    'W_opt' = W_opt,
    'inf_crit_vec' = inf_crit_vec,
    'converge' = converge,
    'weight_matrix' = weight_matrix))
}



#' nn_model_sel_local
#'
#'
#' @param X input matrix
#' @param Y output vector
#' @param q number of hidden units
#' @param W weight vector
#' @param step step size in either direction
#' @param n_iter number of iterations/tracks
#' @param inf_crit information criterion (BIC only)
#' @param unif uniform distribution
#' @return nn_fit_tracks
#' @export
nn_model_sel_local <- function (X, Y, q, W, step = 1, n_iter = 1,
                                inf_crit = 'BIC', unif = 3, ...) {
  # Written: 09/03/2022
  # Last Modified: 09/03/2022
  # Purpose: Performs model selection by only considering q - 1, q, and q + 1
  # Description: Used in combination with variable selection. It allows search
  #              of model selection to consider one larger and one smaller for q.


  df <- data.frame(cbind(X, Y)) #create dataframe of X and Y

  n_candidates <- 2*step + 1 # number of candidate q for models

  inf_crit_matrix <- matrix(NA, nrow = n_candidates, ncol = n_iter)
  rownames(inf_crit_matrix) <- as.character((q - step):(q + step))

  n = nrow(X) # sample size
  p = ncol(X) # number of covariates

  W_opt = vector(mode = "list", length = n_candidates)
  names(W_opt) <- as.character((q - step):(q + step))

  converge = matrix(NA, nrow = n_candidates, ncol = n_iter)
  rownames(converge) <- as.character((q - step):(q + step))

  for (Q in c(q:(q - step), (q + 1):(q + step))) {

    k = (p + 1)*Q + (Q + 1)

    if (Q == q){

      weight_matrix_init <- matrix(rep(W, n_iter), nrow = n_iter, byrow = T) +
        runif(k*n_iter, min = -unif/2, max = unif/2)

      nn <- nn_fit_tracks(X, Y, Q, W_mat = weight_matrix_init,
                          inf_crit = inf_crit, ...)

      nn_q <- nn # store this output to be used for q + 1

      W_opt[[as.character(Q)]] <- nn$W_opt

      inf_crit_matrix[as.character(Q), ] <- nn$inf_crit_vec

      converge[as.character(Q), ] <- nn$converge

    } else if (Q < q & Q > 0){

      weight_matrix <- nn$weight_matrix

      remove_node = apply(X = weight_matrix, 1, remove_unit, dataX = X, Y = Y,
                          q = Q + 1, inf_crit = inf_crit)

      weight_matrix_r = matrix(NA, nrow = n_iter, ncol = k)

      for(f in 1:n_iter){
        unit_to_remove = remove_node[f]
        weight_matrix_r[f,] = weight_matrix[f,][
          -c(((unit_to_remove - 1)*(p + 1) + 1):(unit_to_remove*(p + 1)),
             ((Q + 1)*(p + 1) + unit_to_remove + 1))]
      }

      weight_matrix_init <- weight_matrix_r

      nn <- nn_fit_tracks(X, Y, Q, W_mat = weight_matrix_init,
                          inf_crit = inf_crit, ...)

      W_opt[[as.character(Q)]] <- nn$W_opt

      inf_crit_matrix[as.character(Q), ] <- nn$inf_crit_vec

      converge[as.character(Q), ] <- nn$converge

    } else if (Q > q){

      if(Q == (q + 1)) nn <- nn_q

      weight_matrix <- nn$weight_matrix

      weight_matrix_init <- cbind(
        matrix(weight_matrix[, c(1:((Q - 1)*(p + 1)))],
               byrow = T, nrow = n_iter),
        matrix(runif(n_iter*(p + 1), min = -unif, max = unif), ncol = (p + 1)),
        matrix(weight_matrix[, c(((Q - 1)*(p + 1) + 1):((Q - 1)*(p + 1) + Q))],
               byrow = T, nrow = n_iter),
        matrix(runif(n_iter, min = -unif, max = unif), ncol = 1))

      nn <- nn_fit_tracks(X, Y, Q, W_mat = weight_matrix_init,
                          inf_crit = inf_crit, ...)

      W_opt[[as.character(Q)]] <- nn$W_opt

      inf_crit_matrix[as.character(Q), ] <- nn$inf_crit_vec

      converge[as.character(Q), ] <- nn$converge

    }
  }
  return(list('matrix' = inf_crit_matrix,
              'min' = apply(inf_crit_matrix, 1, min),
              'weights_min' = W_opt,
              'which_min' = as.numeric(names(which.min(apply(inf_crit_matrix, 1, min)))),
              "W_opt" = W_opt[[names(which.min(apply(inf_crit_matrix, 1, min)))]],
              'convergence' = converge,
              'type' = 'local'))
}
