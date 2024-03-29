% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nn_model_sel_local.R
\name{nn_model_sel_local}
\alias{nn_model_sel_local}
\title{nn_model_sel_local}
\usage{
nn_model_sel_local(
  X,
  Y,
  q,
  W,
  step = 1,
  n_iter = 1,
  inf_crit = "BIC",
  unif = 3,
  ...
)
}
\arguments{
\item{X}{input matrix}

\item{Y}{output vector}

\item{q}{number of hidden units}

\item{W}{weight vector}

\item{step}{step size in either direction}

\item{n_iter}{number of iterations/tracks}

\item{inf_crit}{information criterion (BIC only)}

\item{unif}{uniform distribution}
}
\value{
nn_fit_tracks
}
\description{
nn_model_sel_local
}
