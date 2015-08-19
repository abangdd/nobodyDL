#ifndef EXPR_H_
#define EXPR_H_

#include <float.h>
#include "xpu.h"

template <typename DT>
struct opplus {
  XPU_CALLABLE_INLINE DT operator() (const DT a, const DT b) const { return a + b; }
  XPU_CALLABLE_INLINE DT identity() { return (DT)0; }
  XPU_CALLABLE_INLINE DT pooling (const DT a, const size_t region) const { return a / region; }
};

template <typename DT>
struct opsub {
  XPU_CALLABLE_INLINE DT operator() (const DT a, const DT b) const { return a - b; }
  XPU_CALLABLE_INLINE DT identity() { return (DT)0; }
};

template <typename DT>
struct opmul {
  XPU_CALLABLE_INLINE DT operator() (const DT a, const DT b) const { return a * b; }
  XPU_CALLABLE_INLINE DT identity() { return (DT)1; }
};

template <typename DT>
struct opdiv {
  XPU_CALLABLE_INLINE DT operator() (const DT a, const DT b) const { return a / b; }
  XPU_CALLABLE_INLINE DT identity() { return (DT)1; }
};

template <typename DT>
struct opmaximum {
  XPU_CALLABLE_INLINE DT operator() (const DT a, const DT b) const { return max (a, b); }
  XPU_CALLABLE_INLINE DT identity() const;
  XPU_CALLABLE_INLINE DT pooling (const DT a, const size_t region) const { return a; }
};

template <typename DT>
struct opminimum {
  XPU_CALLABLE_INLINE DT operator() (const DT a, const DT b) const { return min (a, b); }
  XPU_CALLABLE_INLINE DT identity() const;
};

template <> XPU_CALLABLE_INLINE float  opmaximum<float >::identity() const { return -FLT_MAX; }
template <> XPU_CALLABLE_INLINE float  opminimum<float >::identity() const { return  FLT_MAX; }
template <> XPU_CALLABLE_INLINE double opmaximum<double>::identity() const { return -DBL_MAX; }
template <> XPU_CALLABLE_INLINE double opminimum<double>::identity() const { return  DBL_MAX; }



template <typename DT>
struct opself {
  XPU_CALLABLE_INLINE DT operator() (const DT a) const { return a; }
};

template <typename DT>
struct opsquare {
  XPU_CALLABLE_INLINE DT operator() (const DT a) const { return a * a; }
};

template <typename DT>
struct opsqrt {
  XPU_CALLABLE_INLINE DT operator() (const DT a) const { return sqrt(a); }
};

template <typename DT>
struct opabs {
  XPU_CALLABLE_INLINE DT operator() (const DT a) const { return fabs(a); }
};

template <typename DT>
struct opexp {
  XPU_CALLABLE_INLINE DT operator() (const DT a) const { return expf(a); }
};

template <typename DT>
struct opinv {
  XPU_CALLABLE_INLINE DT operator() (const DT a) const { return (DT)1 / a; }
};



template <typename DT>
struct equalto {
  XPU_CALLABLE_INLINE static void save (DT &a, DT b) { a  = b;  }
//inline static default_real_t AlphaBLAS(void) { return 1.f; }
//inline static default_real_t BetaBLAS (void) { return 0.f; }
};

template <typename DT>
struct plusto {
  XPU_CALLABLE_INLINE static void save (DT &a, DT b) { a += b;  }
};

template <typename DT>
struct subto {
  XPU_CALLABLE_INLINE static void save (DT &a, DT b) { a -= b;  }
};

template <typename DT>
struct multo {
  XPU_CALLABLE_INLINE static void save (DT &a, DT b) { a *= b;  }
};

template <typename DT>
struct divto {
  XPU_CALLABLE_INLINE static void save (DT& a, DT b) { a /= b;  }
};

#endif
