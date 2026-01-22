#ifndef PGDRAW_INTERNAL_H
#define PGDRAW_INTERNAL_H

#include <Rcpp.h>

/**
 * Internal bridge to the Polya-Gamma sampler.
 * This calls the R function 'pgdraw' from the pgdraw package.
 * By wrapping it this way, we can still parallelize the outer loops in C++.
 */
inline double pgdraw_internal(double b, double z) {
  // Get the pgdraw function from the namespace
  Rcpp::Environment pkg = Rcpp::Environment::namespace_env("pgdraw");
  Rcpp::Function f = pkg["pgdraw"];
  return Rcpp::as<double>(f(b, z));
}

#endif
