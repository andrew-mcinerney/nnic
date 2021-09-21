#' Profile likelihood: function to find optimal weights when fixing a parameter
#'
#' @param X Data
#' @param Y Response
#' @param W Weight vector
#' @param q number of hidden units
#' @param ind index of weight for profiling
#' @param val value for profiling
#' @return Profile likelihood weights
#' @export
prof_likelihood = function(X, Y, W, q, ind, val){

  #Finds optimal W (length k-1) for given weight and value
  opt = optim(par = W, fn = prof_likelihood_pred, method = 'BFGS',
              X = X,Y = Y, q = q, ind = ind, val = val)
  optW = opt$par
  #Adds the predefined weight into the weight vector
  new = c(optW,val)
  id  = c(seq_along(optW), ind-0.5)
  OptW = new[order(id)]
  #Calculates network output
  pred = nn_pred(X, OptW, q)

  return(list('OptW' = OptW, 'logl' = -opt$value, 'pred' = pred))
}


#' Profile likelihood: function to calculate likelihood when parameter is fixed
#'
#' @param X Data
#' @param Y Response
#' @param W Weight vector
#' @param q number of hidden units
#' @param ind index of weight for profiling
#' @param val value for profiling
#' @return Profile likelihood result
#' @export
prof_likelihood_pred = function(X, Y, W, q, ind, val){
  n = nrow(X)
  p = ncol(X)

  new = c(W, val)
  id  = c(seq_along(W), ind - 0.5)
  W_new = new[order(id)]

  if(length(W_new) == ((p + 2)*q + 1)){
    X = cbind(rep(1, n), X)
    h_input = X %*% t(matrix(W_new[1:((p + 1)*q)], nrow = q, byrow = T))

    h_act = cbind(rep(1, ), sigmoid(h_input))
    y_hat = h_act %*% matrix(W_new[c((length(W_new) - q):length(W_new))], ncol=1)

    SSE = sum((y_hat - Y)^2)
    sigma2 = SSE/n

    return(-((-n/2)*log(2*pi*sigma2) - SSE/(2*sigma2)))
  }else{
    return(print('Error: Incorrect number of weights for NN structure'))
  }
}


#' @export
prof_likelihood_ci = function(X, Y, W, q, alpha=0.05, width=0.4, by=0.04){
  k = (ncol(X) + 2)*q + 1
  pred = nn_pred(X, W, q)
  SSE = sum((Y - pred)^2)
  sigma2 = SSE/nrow(X)
  log_likelihood = (-nrow(X)/2)*log(2*pi*sigma2) - SSE/(2*sigma2)
  ts = log_likelihood - qchisq((1 - alpha), 1)/2

  ci = matrix(NA, nrow = k, ncol = 2)
  for(l in 1:k){
    result = c()
    left = seq(W[l] - width/2, W[l], by = by)
    right = seq(W[l], W[l] + width/2, by = by)
    W_pl = W[-l]

    for(j in rev(left)){
      if(all.equal(W[l],j) == T) pl = prof_likelihood(X = X, Y = Y, W = W_pl, q = q,
                                                      ind = l, val = j)
      else pl = prof_likelihood(X = X,Y = Y, W = pl$OptW[-l], q = q,
                                ind = l, val = j)
      result = c(result, pl$logl)
    }
    result = rev(result)
    for(j in right){
      if(all.equal(W[l], j) == T) pl = prof_likelihood(X = X, Y = Y, W = W_pl, q = q,
                                                       ind = l, val = j)
      else pl = prof_likelihood(X = X, Y = Y, W = pl$OptW[-l], q = q,
                                ind = l, val = j)
      result = c(result, pl$logl)
    }
    result = result[-length(result)/2]
    x = c(left, right[-1])

    dist = ((result-rep(ts, (width/by + 1)))^2)
    lin_int1 = approx(x[dist %in% sort(dist)[1:4]][1:2], result[dist %in% sort(dist)[1:4]][1:2])
    dist1 = ((lin_int1$y - rep(ts, 50))^2)
    ci[l, 1] = lin_int1$x[which.min(dist1)]

    lin_int2 = approx(x[dist %in% sort(dist)[1:4]][3:4], result[dist %in% sort(dist)[1:4]][3:4])
    dist2 = ((lin_int2$y - rep(ts, 50))^2)
    ci[k, 2] = lin_int2$x[which.min(dist2)]
  }
  return('ci' = ci)
}
