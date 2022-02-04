#' Variable Importance
#'
#'
#' @param ind Variable index
#' @param n_iter number of iteration
#' @param W weight vector
#' @param q number of hidden units
#' @param X input matrix
#' @param Y output vector
#' @param unif uniform distribution
#' @return Variable importance
#' @export
var_imp = function(X, Y, ind, n_iter, W, q, unif = 1, ...){

  df = as.data.frame(cbind(X[,-ind], Y)) # create dataframe without ind column
  colnames(df)[ncol(df)] = 'Y'
  n = nrow(X)
  inf_crit_vec = rep(NA, n_iter) # vector to store BIC for each iteration

  k = (ncol(X))*q + (q + 1)
  remove_vec = rep(NA, q) # stores which weights to set to zero for the input unit ind
  for(j in 1:q){
    remove_vec[j] = (j - 1)*(ncol(X) + 1) + 1 + ind
  }

  W_new = W[-remove_vec] # removes weights for ind

  weight_matrix_init = matrix(rep(W_new, n_iter), nrow = n_iter, byrow = T) + runif(k*n_iter, min = -unif/2, max = unif/2)

  weight_matrix = matrix(rep(NA,n*k), ncol = k)

  log_likelihood = rep(NA, n_iter)

  for(i in 1:n_iter){
    nn_model =  nnet::nnet(Y~., data = df, size = q, trace = F, linout = T, Wts = weight_matrix_init[i,], ...)
    weight_matrix[i,] = nn_model$wts

    RSS = nn_model$value
    sigma2 = RSS/n
    log_likelihood[i] = (-n/2)*log(2*pi*sigma2) - RSS/(2*sigma2)
    inf_crit_vec[i] = log(n)*(k + 1) - 2*log_likelihood[i]
  }

  full_model = nnic::nn_pred(X, W, q)
  k_full = (ncol(X) + 1)*q + (q + 1)
  RSS = sum((df$Y - full_model)^2)
  sigma2 = RSS/n
  log_likelihood_full = (-n/2)*log(2*pi*sigma2) - RSS/(2*sigma2)
  inf_crit_full = log(n)*(k_full + 1) - 2*log_likelihood_full

  likelihood_ratio = -2*(max(log_likelihood) - log_likelihood_full)
  deg_freedom = k_full - k
  p_value = pchisq(likelihood_ratio, deg_freedom, lower.tail = F)

  return(list('inf_crit' = inf_crit_vec, 'inf_crit_min' = min(inf_crit_vec),
              'BIC_full' = inf_crit_full, 'loglik' = max(log_likelihood),
              'loglik_full' = log_likelihood_full, 'df' = deg_freedom,
              'likelihood_ratio' = likelihood_ratio, 'p_val' = p_value,
              'W_opt' = weight_matrix[which.min(inf_crit_vec),]))
}

#' Effect of input
#'
#'
#' @param i Variable index
#' @param W weight vector
#' @param h number of hidden units
#' @param X input matrix
#' @param val value of input units
#' @param length length of vector to be calculated
#' @param range range of values to be computed
#' @return Effect of variable
#' @export
nn_effect = function(X, ind, W, q, val = rep(0, ncol(X)), length = 100, range = c(-3, 3), sigma = 0){

  val = val[-ind]
  X_ind = seq(from = range[1], to = range[2], length.out = length)
  X_val = matrix(rep(val, length), nrow = length, byrow = T)

  new = cbind(X_val, X_ind)
  id = c(1:ncol(X_val), ind - 0.5)
  X_new = new[, order(id)]

  pred = nn_pred(X_new, W, q) + rnorm(length, 0, sigma)

  return(list('x' = X_ind, 'pred' = pred))
}


#' Variable Selection
#'
#'
#' @param X input matrix
#' @param Y output vector
#' @param nn output from nn_model_sel function
#' @param n_iter number of iteration
#' @return Variable Selection
#' @export
nn_variable_sel <- function(X, Y, nn, n_iter, ...){
  p = ncol(X)

  continue = TRUE

  W_opt = nn$weights_min[[nn$which_min]]

  dropped = c()

  while(continue == TRUE){
    input_BIC = rep(NA, p) #store BIC with each input unit removed
    for(i in 1:p){
      var_imp_nn <- var_imp(X, Y, ind = i, n_iter = n_iter,
                            W = W_opt,
                            q = nn$which_min, ...)

      input_BIC[i] <- var_imp_nn$inf_crit_min
    }

    if(min(input_BIC) <= nn$min[nn$which_min]){
      X = X[,-which.min(input_BIC)] #drop irrelevant input
      p = ncol(X)
      W_opt = var_imp_nn$W_opt
      dropped = c(dropped, which.min(input_BIC))
    }else{
      continue = FALSE
    }
  }
  return(list('X' = X, 'p' = p, 'W_opt' = W_opt, 'dropped' = dropped,
              'full_BIC' = nn$min[nn$which_min], 'BIC' = min(input_BIC)))
}
