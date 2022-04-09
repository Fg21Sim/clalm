# distutils: language=c++
#
#  This file is part of Healpy.
#
#  Healpy is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  Healpy is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Healpy; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
#  For more information about Healpy, see http://code.google.com/p/healpy
#
import numpy  as np
import cython
cimport numpy as np
cimport libc
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.math cimport sqrt, floor, fabs
from libcpp cimport bool as cbool

def alm_getlmmax(a, lmax, mmax):
    if lmax is None:
        if mmax is None:
            lmax = alm_getlmax(a.size)
            mmax = lmax
        else:
            lmax = alm_getlmax2(a.size, mmax)
    elif mmax is None:
        mmax = lmax
    return lmax, mmax


@cython.cdivision(True)
cdef inline long alm_getlmax(long s):
    cdef double x
    x=(-3+np.sqrt(1.+8.*s))/2
    if x != floor(x):
        return -1
    else:
        return <long>floor(x)


@cython.cdivision(True)
cdef inline long alm_getlmax2(long s, long mmax):
    cdef double x
    x = (2 * s + mmax ** 2 - mmax - 2.) / (2 * mmax + 2.)
    if x != floor(x):
        return -1
    else:
        return <long>floor(x)


@cython.cdivision(True)
cdef inline long alm_getidx(long lmax, long l, long m):
    return m*(2*lmax+1-m)/2+l

def alm2cl(alms, alms2 = None, lmax = None, mmax = None, lmax_out = None):
    cdef long Nspec, Nspec2
    if not hasattr(alms, '__len__'):
        raise ValueError('alms must be an array or a sequence of arrays')
    if not hasattr(alms[0], '__len__'):
        alms_lonely = True
        alms = [alms]
    else:
        alms_lonely = False

    Nspec = len(alms)

    if alms2 is None:
        alms2 = alms

    if not hasattr(alms2, '__len__'):
        raise ValueError('alms2 must be an array or a sequence of arrays')
    if not hasattr(alms2[0], '__len__'):
        alms2 = [alms2]
    Nspec2 = len(alms2)

    if Nspec != Nspec2:
        raise ValueError('alms and alms2 must have same number of spectra')

    ##############################################
    # Check sizes of alm's and lmax/mmax/lmax_out
    #
    cdef long almsize
    almsize = alms[0].size
    for i in xrange(Nspec):
        if alms[i].size != almsize or alms2[i].size != almsize:
            raise ValueError('all alms must have same size')

    lmax, mmax = alm_getlmmax(alms[0], lmax, mmax)

    if lmax_out is None:
        lmax_out = lmax


    #######################
    # Computing the spectra
    #
    cdef long j, l, m, limit
    cdef long lmax_ = lmax, mmax_ = mmax
    cdef long lmax_out_ = lmax_out

    cdef np.ndarray[double, ndim=1] powspec_
    cdef np.ndarray[np.complex128_t, ndim=1] alm1_
    cdef np.ndarray[np.complex128_t, ndim=1] alm2_

    spectra = []
    for n in xrange(Nspec): # diagonal rank
        for m in xrange(0, Nspec - n): # position in the diagonal
            powspec_ = np.zeros(lmax + 1)
            alm1_ = alms[m]
            alm2_ = alms2[m + n]
            # compute cross-spectrum alm1[n] x alm2[n+m]
            # and place result in result list
            for l in range(lmax_ + 1):
                j = alm_getidx(lmax_, l, 0)
                powspec_[l] = alm1_[j].real * alm2_[j].real
                limit = l if l <= mmax else mmax
                for m in range(1, limit + 1):
                    j = alm_getidx(lmax_, l, m)
                    powspec_[l] += 2 * (alm1_[j].real * alm2_[j].real +
                                        alm1_[j].imag * alm2_[j].imag)
                powspec_[l] /= (2 * l + 1)
            spectra.append(powspec_[:lmax_out+1])

    # if only one alm was given, returns only cl and not a list with one cl
    if alms_lonely:
        spectra = spectra[0]

    return np.array(spectra)


