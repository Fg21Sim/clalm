/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file math_utils.h
 *  Various convenience mathematical functions.
 *
 *  Copyright (C) 2002-2015 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef PLANCK_MATH_UTILS_H
#define PLANCK_MATH_UTILS_H

#include <cmath>
#include <vector>
#include <algorithm>

/*! Helper function for linear interpolation (or extrapolation).
    The array must be ordered in ascending order; no two values may be equal. */
template<typename T, typename Iter, typename Comp> inline void interpol_helper
  (const Iter &begin, const Iter &end, const T &val, Comp comp, tsize &idx,
  T &frac)
  {
  using namespace std;
  planck_assert((end-begin)>1,"sequence too small for interpolation");
  idx = lower_bound(begin,end,val,comp)-begin;
  if (idx>0) --idx;
  idx = min(tsize(end-begin-2),idx);
  frac = (val-begin[idx])/(begin[idx+1]-begin[idx]);
  }

#endif
