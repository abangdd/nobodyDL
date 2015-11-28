#ifndef EXPR_H_
#define EXPR_H_

#include <float.h>
#include "xpu.h"

template <typename DT>
struct opplus {
  XPU_INLINE DT operator() (const DT a, const DT b) const { return a + b; }
  XPU_INLINE DT identity() { return (DT)0; }
  XPU_INLINE DT pooling (const DT a, const size_t region) const { return a / region; }
};

template <typename DT>
struct opsub {
  XPU_INLINE DT operator() (const DT a, const DT b) const { return a - b; }
  XPU_INLINE DT identity() { return (DT)0; }
};

template <typename DT>
struct opmul {
  XPU_INLINE DT operator() (const DT a, const DT b) const { return a * b; }
  XPU_INLINE DT identity() { return (DT)1; }
};

template <typename DT>
struct opdiv {
  XPU_INLINE DT operator() (const DT a, const DT b) const { return a / b; }
  XPU_INLINE DT identity() { return (DT)1; }
};

template <typename DT>
struct opmaximum {
  XPU_INLINE DT operator() (const DT a, const DT b) const { return max (a, b); }
  XPU_INLINE DT identity() const;
  XPU_INLINE DT pooling (const DT a, const size_t region) const { return a; }
};

template <typename DT>
struct opminimum {
  XPU_INLINE DT operator() (const DT a, const DT b) const { return min (a, b); }
  XPU_INLINE DT identity() const;
};

template <> XPU_INLINE float  opmaximum<float >::identity() const { return -FLT_MAX; }
template <> XPU_INLINE float  opminimum<float >::identity() const { return  FLT_MAX; }
template <> XPU_INLINE double opmaximum<double>::identity() const { return -DBL_MAX; }
template <> XPU_INLINE double opminimum<double>::identity() const { return  DBL_MAX; }



template <typename DT>
struct opself {
  XPU_INLINE DT operator() (const DT a) const { return a; }
};

template <typename DT>
struct opsquare {
  XPU_INLINE DT operator() (const DT a) const { return a * a; }
};

template <typename DT>
struct opsqrt {
  XPU_INLINE DT operator() (const DT a) const { return sqrt(a); }
};

template <typename DT>
struct opabs {
  XPU_INLINE DT operator() (const DT a) const { return fabs(a); }
};

template <typename DT>
struct opexp {
  XPU_INLINE DT operator() (const DT a) const { return expf(a); }
};

template <typename DT>
struct opinv {
  XPU_INLINE DT operator() (const DT a) const { return (DT)1 / a; }
};



template <typename DT>
struct equalto {
  XPU_INLINE static void save (DT &a, DT b) { a  = b;  }
//inline static default_real_t AlphaBLAS(void) { return 1.f; }
//inline static default_real_t BetaBLAS (void) { return 0.f; }
};

template <typename DT>
struct plusto {
  XPU_INLINE static void save (DT &a, DT b) { a += b;  }
};

template <typename DT>
struct subto {
  XPU_INLINE static void save (DT &a, DT b) { a -= b;  }
};

template <typename DT>
struct multo {
  XPU_INLINE static void save (DT &a, DT b) { a *= b;  }
};

template <typename DT>
struct divto {
  XPU_INLINE static void save (DT& a, DT b) { a /= b;  }
};

#endif
