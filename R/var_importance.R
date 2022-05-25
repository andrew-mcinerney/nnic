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
var_imp = function(X, Y, ind, n_iter, W = NULL, W_mat = NULL, q, unif = 3,
                   inf_crit = 'BIC', dev = unif/2, ...){

  df = as.data.frame(cbind(X[,-ind], Y)) # create dataframe without ind column
  colnames(df)[ncol(df)] = 'Y'
  n = nrow(X)
  inf_crit_vec = rep(NA, n_iter) # vector to store BIC for each iteration

  k = (ncol(X))*q + (q + 1)
  remove_vec = rep(NA, q) # stores which weights to set to zero for the input unit ind
  for(j in 1:q){
    remove_vec[j] = (j - 1)*(ncol(X) + 1) + 1 + ind
  }

  if (!is.null(W_mat)){
    W_mat_new = matrix(W_mat[, -remove_vec], nrow = n_iter) # removes weights for ind

    weight_matrix_init = W_mat_new

  } else if (!is.null(W)){
    W_new = W[-remove_vec] # removes weights for ind

    weight_matrix_init = matrix(rep(W_new, n_iter), nrow = n_iter, byrow = T) + runif(k*n_iter, min = -dev, max = dev)
  } else {
    stop("Either W or W_mat must not be null")
  }

  weight_matrix = matrix(rep(NA,n*k), ncol = k)

  log_likelihood = rep(NA, n_iter)

  for(i in 1:n_iter){
    nn_model =  nnet::nnet(Y~., data = df, size = q, trace = F, linout = T, Wts = weight_matrix_init[i,], ...)
    weight_matrix[i,] = nn_model$wts

    RSS = nn_model$value
    sigma2 = RSS/n
    log_likelihood[i] = (-n/2)*log(2*pi*sigma2) - RSS/(2*sigma2)
    inf_crit_vec[i] = ifelse(inf_crit == 'AIC', (2*(k+1) - 2*log_likelihood[i]),
                             ifelse(inf_crit == 'BIC', (log(n)*(k+1) - 2*log_likelihood[i]),
                                    ifelse(inf_crit == 'AICc', (2*(k+1)*(n/(n-(k+1)-1)) - 2*log_likelihood[i]),NA)))
  }

  full_model = nnic::nn_pred(X, W, q)
  k_full = (ncol(X) + 1)*q + (q + 1)
  RSS = sum((df$Y - full_model)^2)
  sigma2 = RSS/n
  log_likelihood_full = (-n/2)*log(2*pi*sigma2) - RSS/(2*sigma2)
  inf_crit_full = ifelse(inf_crit == 'AIC', (2*(k_full+1) - 2*log_likelihood_full),
                         ifelse(inf_crit == 'BIC', (log(n)*(k_full+1) - 2*log_likelihood_full),
                                ifelse(inf_crit == 'AICc', (2*(k_full+1)*(n/(n-(k_full+1)-1)) - 2*log_likelihood_full),NA)))

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
nn_variable_sel <- function(X, Y, n_iter, q = NULL, nn = NULL, unif = 3,
                            inf_crit = 'BIC', dev = unif/2, ...){
  p <- ncol(X)

  n <- nrow(X)

  continue <- TRUE

  if (!is.null(nn)){
    W_opt <- nn$W_opt

    # if statement to fix error when using local method and uses name not index
    if (is.null(nn$type)) {
      min_BIC <- nn$min[nn$which_min]
    } else if (nn$type == 'local') {
      min_BIC <- nn$min[as.character(nn$which_min)]
    }

    full_BIC <- min_BIC
    q <- nn$which_min
    k = (p + 2)*q + 1

    W_opt_mat <- matrix(rep(W_opt, n_iter), nrow = n_iter, byrow = T) +
      runif(k*n_iter, min = -dev, max = dev)

  } else if (!is.null(q)){
    df = as.data.frame(cbind(X, Y))
    colnames(df)[ncol(df)] = 'Y'

    k <- (p+2)*q + 1
    W <- matrix(runif(n_iter*k, min = -unif, max = unif),
                ncol = k)
    weight_matrix <- matrix(rep(NA, n_iter*k),ncol=k)
    inf_crit_vec <- rep(NA, n_iter)

    for(iter in 1:n_iter){
      nn_model <-  nnet::nnet(Y~., data = df, size = q, trace = F, linout = T,
                              Wts = W[iter,], ...)
      weight_matrix[iter,] <- nn_model$wts

      sigma2 = nn_model$value/n
      log_likelihood = (-n/2)*log(2*pi*sigma2) - nn_model$value/(2*sigma2)
      inf_crit_vec[iter] = ifelse(inf_crit == 'AIC', (2*(k+1) - 2*log_likelihood),
                                  ifelse(inf_crit == 'BIC', (log(n)*(k+1) - 2*log_likelihood),
                                         ifelse(inf_crit == 'AICc', (2*(k+1)*(n/(n-(k+1)-1)) - 2*log_likelihood),NA)))
    }
    W_opt <- weight_matrix[which.min(inf_crit_vec),]
    W_opt_mat <- matrix(rep(W_opt, n_iter), nrow = n_iter, byrow = T) +
      runif(k*n_iter, min = -dev, max = dev)
    min_BIC <- min(inf_crit_vec)
    full_BIC <- min_BIC
  } else {
    stop('No inital network or hidden layer size supplied')
  }

  colnames(X) = 1:p

  X_full <- X

  dropped <- c()

  while(continue == TRUE){
    input_BIC = rep(NA, p) #store BIC with each input unit removed
    var_imp_nn <- vector(mode = 'list', length = p)
    for(i in 1:p){
      var_imp_nn[[i]] <- var_imp(as.matrix(X), Y, ind = i, n_iter = n_iter,
                                 W_mat = W_opt_mat,
                                 q = q, inf_crit = inf_crit, unif = unif,
                                 dev = dev,
                                 ...)

      input_BIC[i] <- var_imp_nn[[i]]$inf_crit_min
    }

    if(min(input_BIC) <= min_BIC & p > 2){
      X = X[,-which.min(input_BIC)] #drop irrelevant input
      p = ncol(as.matrix(X))
      k = (p + 2)*q + 1

      W_opt = var_imp_nn[[which.min(input_BIC)]]$W_opt
      W_opt_mat <- matrix(rep(W_opt, n_iter), nrow = n_iter, byrow = T) +
        runif(k*n_iter, min = -dev, max = dev)

      dropped <- colnames(X_full)[!colnames(X_full) %in% colnames(X)]
      min_BIC <- min(input_BIC)
    } else if (min(input_BIC) <= min_BIC & p <= 2 & p > 1){
      col_names <- colnames(X)[-which.min(input_BIC)] # store column name as X will become vector
      X = X[,-which.min(input_BIC)] #drop irrelevant input
      p = ncol(as.matrix(X))
      k = (p + 2)*q + 1

      W_opt = var_imp_nn[[which.min(input_BIC)]]$W_opt
      W_opt_mat <- matrix(rep(W_opt, n_iter), nrow = n_iter, byrow = T) +
        runif(k*n_iter, min = -dev, max = dev)

      dropped <- colnames(X_full)[!colnames(X_full) %in% col_names]
      min_BIC <- min(input_BIC)
      continue <- FALSE
    } else{
      continue <- FALSE
    }
  }
  return(list('X' = X, 'p' = p, 'W_opt' = W_opt, 'dropped' = dropped,
              'full_BIC' = full_BIC, 'BIC' = min_BIC))
}
