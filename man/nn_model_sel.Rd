% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nn_model_sel.R
\name{nn_model_sel}
\alias{nn_model_sel}
\title{Model selection for a MLP with a single hidden layer.}
\usage{
nn_model_sel(
  X,
  Y,
  q_max,
  q_min = 1,
  W = NULL,
  n_iter = 1,
  inf_crit = "BIC",
  unif = 3,
  method = "top_down",
  remove = "best",
  plot = F,
  dev = unif/2,
  ...
)
}
\arguments{
\item{X}{Matrix of inputs}

\item{Y}{Output vector}

\item{q_max}{Largest number of hidden units to consider}

\item{n_iter}{Number of iterations}

\item{inf_crit}{Information criterion}

\item{unif}{Uniform distribution}

\item{method}{Procedure}
}
\value{
Optimal number of hidden units
}
\description{
This function determines the optimal number of hidden units to use in a
single-hidden-layer MLP. It allows for a bottom-up and top-down approach.
}
