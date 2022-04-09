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
import numpy as np
import synalm

def getsize(lmax, mmax=None):
    if mmax is None or mmax < 0 or mmax > lmax:
        mmax = lmax
    return mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1

def cl2alm(cls, lmax=None, mmax=None, new=False, verbose=True):
    if lmax is None or lmax < 0:
        lmax = len(cls) - 1
    if mmax is None or mmax < 0:
        mmax = lmax
    cls_list = [np.asarray(cls, dtype=np.float64)]
    szalm = getsize(lmax, mmax)
    alm = np.zeros(szalm, "D")
    alm.real = np.random.standard_normal(szalm)
    alm.imag = np.random.standard_normal(szalm)
    alms_list = [alm]
    synalm._synalm(cls_list, alms_list, lmax, mmax)
    return alm
