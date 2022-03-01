#' Model selection for a MLP with a single hidden layer.
#'
#' This function determines the optimal number of hidden units to use in a
#' single-hidden-layer MLP. It allows for a bottom-up and top-down approach.
#'
#'
#' @param q_max Largest number of hidden units to consider
#' @param n_iter Number of iterations
#' @param X Matrix of inputs
#' @param Y Output vector
#' @param inf_crit Information criterion
#' @param unif Uniform distribution
#' @param method Procedure
#' @return Optimal number of hidden units
#' @export
# nn_model_sel, but takes nn_var_sel as input

nn_model_sel = function(X, Y, q_max, q_min = 1, W = NULL, n_iter = 1, inf_crit = 'BIC', unif = 3,
                        method = 'top_down', remove = 'best', plot = F, ...){

  df <- as.data.frame(cbind(X, Y)) #create dataframe of X and Y
  colnames(df)[ncol(df)] <- 'Y'

  inf_crit_matrix <- matrix(NA, nrow = (q_max - q_min + 1), ncol = n_iter) #store information criteria
  rownames(inf_crit_matrix) <- as.character(q_min:q_max)

  n = nrow(X) # sample size
  p = ncol(X) # number of covariates

  W_opt = vector(mode = "list", length = (q_max - q_min + 1))

  converge = matrix(NA, nrow = (q_max - q_min + 1), ncol = n_iter)

  if(method == 'bottom_up'){

    for(q in q_min:q_max){
      k = (p + 1)*q + (q + 1)

      if(q == q_min){
        weight_matrix_init = matrix(runif(n_iter*k, min = -unif, max = unif),
                                    ncol = k)
      }else{
        weight_matrix_init = cbind(matrix(weight_matrix[, c(1:((q - 1)*(p + 1)))], byrow = T, nrow = n_iter),
                                   matrix(runif(n_iter*(p + 1), min = -unif, max = unif), ncol = (p + 1)),
                                   matrix(weight_matrix[, c(((q - 1)*(p + 1) + 1):((q - 1)*(p + 1) + q))], byrow = T, nrow = n_iter),
                                   matrix(runif(n_iter, min = -unif, max = unif), ncol = 1))
      }

      weight_matrix = matrix(rep(NA, n_iter*k),ncol=k)

      for(iter in 1:n_iter){
        nn_model =  nnet::nnet(Y~., data = df, size = q, trace = FALSE,
                               linout = T, Wts = weight_matrix_init[iter,], ...)

        weight_matrix[iter,] = nn_model$wts

        SSE = sum((df$Y - nn_model$fitted.values)^2)
        sigma2 = SSE/n
        log_likelihood = (-n/2)*log(2*pi*sigma2) - SSE/(2*sigma2)

        inf_crit_matrix[q, iter] = ifelse(inf_crit == 'AIC', (2*(k+1) - 2*log_likelihood),
                                          ifelse(inf_crit == 'BIC', (log(n)*(k+1) - 2*log_likelihood),
                                                 ifelse(inf_crit == 'AICc', (2*(k+1)*(n/(n - (k+1) - 1)) - 2*log_likelihood),NA)))
        converge[q, iter] = nn_model$convergence
      }
      W_opt[[q]] = weight_matrix[which.min(inf_crit_matrix[q,]),]
    }
  } else if(method == 'top_down'){

    for(q in q_max:q_min){
      k = (p + 1)*q + (q + 1)

      if (q == q_max & is.null(W)){
        weight_matrix_init <- matrix(runif(n_iter*k, min = -unif, max = unif),
                                     ncol = k)
      } else if (q == q_max & !is.null(W)){
        weight_matrix_init <- matrix(rep(W, n_iter), nrow = n_iter, byrow = T) +
          runif(k*n_iter, min = -unif/2, max = unif/2)
      } else {
        weight_matrix_init <- weight_matrix_r
      }

      weight_matrix = matrix(rep(NA, n_iter*k), ncol = k)

      for(iter in 1:n_iter){
        nn_model =  nnet::nnet(Y~., data = df, size = q, trace = F, linout = T,
                               Wts=weight_matrix_init[iter,], ...)
        weight_matrix[iter,] = nn_model$wts

        SSE = sum((df$Y - nn_model$fitted.values)^2)
        sigma2 = SSE/n
        log_likelihood = (-n/2)*log(2*pi*sigma2) - SSE/(2*sigma2)
        inf_crit_matrix[q, iter] = ifelse(inf_crit == 'AIC', (2*(k+1) - 2*log_likelihood),
                                          ifelse(inf_crit == 'BIC', (log(n)*(k+1) - 2*log_likelihood),
                                                 ifelse(inf_crit == 'AICc', (2*(k+1)*(n/(n-(k+1)-1)) - 2*log_likelihood),NA)))
        converge[q, iter] = nn_model$convergence
      }
      W_opt[[q]] = weight_matrix[which.min(inf_crit_matrix[q,]),]

      if(remove == 'best'){
        remove_node = apply(X = weight_matrix, 1, remove_unit, dataX = X, Y = Y, q = q, inf_crit = inf_crit)
      }else if(remove == 'random'){
        remove_node = sample(1:q, size = n_iter, replace = T)
      }

      weight_matrix_r = matrix(NA, nrow = n_iter, ncol = k - p - 2)

      for(f in 1:n_iter){
        unit_to_remove = remove_node[f]
        weight_matrix_r[f,] = weight_matrix[f,][-c(((unit_to_remove - 1)*(p + 1) + 1):(unit_to_remove*(p + 1)),(q*(p + 1) + unit_to_remove + 1))]
      }
    }
  }else{
    print('Error: Method not valid')
  }

  #plotting inf_crit_matrix using ggplot2
  if(plot == TRUE){
    inf_crit_df <- as.data.frame(inf_crit_matrix)
    inf_crit_df$id <- 1:nrow(inf_crit_df)
    colnames(inf_crit_df) = c(as.character(1:n_iter), 'id')
    plot_df <- reshape2::melt(inf_crit_df, id.var = "id")
    colnames(plot_df)[2] = 'n_iter'

    p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = id, y = value, group = n_iter, colour = n_iter))  +
      ggplot2::geom_point() +
      ggplot2::geom_line(ggplot2::aes(lty = n_iter)) + ggplot2::labs(x = 'Number of Hidden Units', y = paste(inf_crit)) +
      ggplot2::scale_x_continuous(breaks = c(q_min:q_max)) + ggplot2::ggtitle(paste0(method, ' approach')) +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), text = ggplot2::element_text(size=17))

    print(p)
  }

  return(list('matrix' = inf_crit_matrix, 'min' = apply(inf_crit_matrix, 1, min),
              'weights_min' = W_opt, 'which_min' = as.numeric(which.min(apply(inf_crit_matrix, 1, min))),
              "W_opt" = W_opt[[as.numeric(which.min(apply(inf_crit_matrix, 1, min)))]],
              'convergence' = converge))
}



#' @export
NN_IC_Group = function(H,N,X,Y,IC='AIC',unif1=1,unif2=unif1/2,method='forward',group_size=1){

  df = as.data.frame(cbind(X,Y))
  colnames(df)[ncol(df)] = 'Y'
  IC.m=matrix(NA,nrow=H,ncol=N)
  n=nrow(X)
  weight_opt = vector(mode = "list", length = H)
  low_weights = vector(mode = 'list', length = H)

  if(method == 'forward'){

    for(h in 1:H){
      K = (ncol(X)+1)*h + (h+1)

      if(h == 1){
        weight_matrix_init = matrix(runif(N*K,min=-unif1,max=unif1),ncol=K)
      }else{
        weight_matrix_init=cbind(matrix(rep(t(low_weights[[h-1]]),(N/group_size)),nrow=N,byrow=T)[,1:((h-1)*(ncol(X)+1))],matrix(runif(N*(ncol(X)+1),min=-unif1,max=unif1),ncol=(ncol(X)+1)),
                                 matrix(rep(t(low_weights[[h-1]]),(N/group_size)),nrow=N,byrow=T)[,((h-1)*(ncol(X)+1)+1):((h-1)*(ncol(X)+1)+h)],matrix(runif(N*1,min=-unif1,max=unif1),ncol=1))
      }

      weight_matrix = matrix(rep(NA,N*K),ncol=K)

      for(i in 1:N){
        nn =  nnet::nnet(Y~.,data=df,size=h,trace=FALSE,linout=T,Wts=weight_matrix_init[i,])
        weight_matrix[i,] = nn$wts

        SSE = sum((df$Y-nn$fitted.values)^2)
        sigma2 = SSE/n
        log_likelihood = (-n/2)*log(2*pi*sigma2) - SSE/(2*sigma2)
        IC.m[h,i] = ifelse(IC=='AIC',(2*K - 2*log_likelihood),
                           ifelse(IC=='BIC',(log(n)*K - 2*log_likelihood),
                                  ifelse(IC=='AICc',(2*K*(n/(n-K-1)) - 2*log_likelihood),NA)))
      }
      weight_opt[[h]] = weight_matrix[which.min(IC.m[h,]),]
      low_weights[[h]] = weight_matrix[order(IC.m[h,])[1:group_size],]
    }
    return(list('matrix'=IC.m,'min'=apply(IC.m,1,min),'weights_min'=weight_opt,'which_min'=which.min(apply(IC.m,1,min))))

  } else if(method == 'backward'){

    for(h in H:1){
      K = (ncol(X)+1)*h + (h+1)

      if(h == H){
        weight_matrix_init = matrix(runif(N*K,min=-unif1,max=unif1),ncol=K)
      }else{
        weight_matrix_init=matrix(rep(t(low_weights.r),(N/group_size)),nrow=N,byrow=T)+runif(N,min=-unif2,max=unif2)
      }

      weight_matrix = matrix(rep(NA,N*K),ncol=K)

      for(i in 1:N){
        nn =  nnet::nnet(Y~.,data=df,size=h,trace=FALSE,linout=T,Wts=weight_matrix_init[i,])
        weight_matrix[i,] = nn$wts

        SSE = sum((df$Y-nn$fitted.values)^2)
        sigma2 = SSE/n
        log_likelihood = (-n/2)*log(2*pi*sigma2) - SSE/(2*sigma2)
        IC.m[h,i] = ifelse(IC=='AIC',(2*K - 2*log_likelihood),
                           ifelse(IC=='BIC',(log(n)*K - 2*log_likelihood),
                                  ifelse(IC=='AICc',(2*K*(n/(n-K-1)) - 2*log_likelihood),NA)))
      }
      weight_opt[[h]] = weight_matrix[which.min(IC.m[h,]),]
      low_weights[[h]] = weight_matrix[order(IC.m[h,])[1:group_size],]
      remove_node = apply(X=low_weights[[h]],1,RemoveNode,unit=h,dataX=X,Y=Y)
      low_weights.r = matrix(NA,nrow=group_size,ncol=K-ncol(X)-2)
      for(f in 1:group_size){
        min_node = remove_node[f]
        low_weights.r[f,] = low_weights[[h]][f,][-c(((min_node-1)*(ncol(X)+1)+1):(min_node*(ncol(X)+1)),(h*(ncol(X)+1)+min_node+1))]
      }
    }
    return(list('matrix'=IC.m,'min'=apply(IC.m,1,min),'weights_min'=weight_opt,'which_min'=which.min(apply(IC.m,1,min))))
  }else{
    print('Error: Method not valid')
  }
}
