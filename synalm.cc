/*
 *  This file is part of Healpy.
 *
 *  Healpy is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Healpy is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Healpy; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about Healpy, see http://code.google.com/p/healpy
 */
/*

   This module provides Healpix functions to Python.
   It uses the healpix_cxx library.

*/

#include <Python.h>

#include "numpy/arrayobject.h"

#include <string>
#include <iostream>

#include "arr.h"
#include "alm.h"
#include "xcomplex.h"

#define IS_DEBUG_ON 0

/* Some helpful macro */
#define XMALLOC(X,Y,Z) if( !(X = (Y*)malloc(Z*sizeof(Y))) ) { PyErr_NoMemory(); goto fail;}
#define XNEW(X,Y,Z) if( !(X = new Y[Z]) ) { PyErr_NoMemory(); goto fail; }
#define XFREE(X) if( X ) free(X);
#define XDELETE(X) if( X ) delete[] X;
#define DBGPRINTF(X,...) if( IS_DEBUG_ON ) printf(X, ## __VA_ARGS__)

static long 
getidx(long n, long i, long j) {
  long tmp;
  if( i > j )
    {tmp = j; j=i; i=tmp;}
  return i*(2*n-1-i)/2+j;
}

static long 
getn(long s) {
  long x;
  x = (long)floor((-1+sqrt(1+8*s))/2);
  if( (x*(x+1)/2) != s )
    return -1;
  else
    return x;
}

static void 
cholesky(int n, double *data, double *res) {
  int i,j,k;
  double sum;

  for( j=0; j<n; j++ )
    {
      for( i=0; i<n; i++ )
        {
          if( i==j )
            {
              sum = data[getidx(n,j,j)];
              for(k=0; k<j; k++ )
                sum -= res[getidx(n,k,j)]*res[getidx(n,k,j)];
              if( sum <= 0 )
                res[getidx(n,j,j)] = 0.0;
              else
                res[getidx(n,j,j)] = sqrt(sum);
            }
          else if( i>j)
            {
              sum = data[getidx(n,i,j)];
              for( k=0; k<j; k++ )
                sum -= res[getidx(n,i,k)]*res[getidx(n,j,k)];
              if( res[getidx(n,j,j)] != 0.0 )
                res[getidx(n,i,j)] = sum/res[getidx(n,j,j)];
              else
                res[getidx(n,i,j)] = 0.0;
            }
        }
    }
  return;
}

static PyObject *
_synalm(PyObject *self, PyObject *args, PyObject *kwds) {
  int lmax=-1, mmax=-1;
  int ncl, nalm;
  const double sqrt_two = sqrt(2.);

  /* Take also a sequence of unit variance random vectors for the alm. */

  static const char* kwlist[] = {"","", "", "", NULL};

  PyObject *t = NULL;
  PyObject *u = NULL;
  DBGPRINTF("Parsing keyword\n");
  if( !PyArg_ParseTupleAndKeywords(args, kwds, "OOii", (char **)kwlist,
                                   &t,
                                   &u,
                                   &lmax, &mmax) )
    return NULL;

  DBGPRINTF("Checking sequence\n");
  if( (!PySequence_Check(t)) || (!PySequence_Check(u)) )
    return NULL;

  ncl = PySequence_Size(t);
  nalm = getn(ncl);
  DBGPRINTF("Sequence size: ncl=%d, nalm=%d\n", ncl, nalm);
  if( nalm<=0 || (PySequence_Size(u)!=nalm) ) {
  	std::cout << "First argument must be a sequence with "
			  << "n(n+1)/2 elements, and second argument "
			  << "a sequence with n elements."
			  << std::endl;
	return NULL;
  }

  DBGPRINTF("Allocating memory\n");
  /* Memory allocation */
  PyArrayObject **cls = NULL;
  PyArrayObject **alms = NULL;
  Alm< xcomplex<double> > *almalms = NULL;
  double *mat = NULL;
  double *res = NULL;
  XMALLOC(cls, PyArrayObject*, ncl);
  XMALLOC(alms, PyArrayObject*, nalm);
  XNEW(almalms, Alm< xcomplex<double> >, nalm);
  XMALLOC(mat, double, ncl);
  XMALLOC(res, double, ncl);

  /*  From now on, I should do a 'goto fail' to return from the function
      in order to free allocated memory */
  /* Get the cls objects.
     If an object is None, set the array to NULL
  */
  for( int i=0; i<ncl; i++ )
    {
      DBGPRINTF("Get item cl %d/%d\n", i+1, ncl);
      PyObject *o;
      o = PySequence_GetItem(t,i);
      /* I decrease reference counts here,
         because PySequence_GetItem increase
         reference count, and I just want to
         borrow a reference the time of this
         function. */
      Py_XDECREF(o);
      if( o == Py_None )
        {
          cls[i] = NULL;
          DBGPRINTF("Cls[%d] is None\n", i);
        }
      else if( ! PyArray_Check(o) )
        {
          PyErr_SetString(PyExc_TypeError,
                          "First argument must be a sequence of "
                          "arrays");
          goto fail;
        }
      else
        cls[i] = (PyArrayObject*) o;
    }
  for( int i=0; i<nalm; i++ )
    {
      PyObject *o;
      DBGPRINTF("Get item alm %d/%d\n", i+1, nalm);
      o = PySequence_GetItem(u,i);
      /* I decrease reference counts here,
         because PySequence_GetItem increase
         reference count, and I just want to
         borrow a reference the time of this
         function. */
      Py_XDECREF(o);
      if( ! PyArray_Check(o) )
        {
          PyErr_SetString(PyExc_TypeError,
                          "First argument must be a sequence of "
                          "arrays");
          goto fail;
        }
      alms[i] = (PyArrayObject*) o;
    }
  if( lmax<0 )
    {
      PyErr_SetString(PyExc_ValueError,
                      "lmax must be positive.");
      goto fail;
    }
  if( mmax <0 || mmax >lmax )
    mmax=lmax;

  DBGPRINTF("lmax=%d, mmax=%d\n", lmax, mmax);

  /* Now, I check the arrays cls and alms are 1D and complex for alms */
  DBGPRINTF("Check dimension and size of cls\n");
  for( int i=0; i<ncl; i++ )
    {
      if( cls[i] == NULL )
        continue;
      if( (cls[i]->nd != 1)
          //|| ((cls[i]->descr->type != 'd') && (cls[i]->descr->type != 'f')) )
          || (cls[i]->descr->type != 'd') )
        {
          PyErr_SetString(PyExc_TypeError,
                      "Type of cls must be float64 and arrays must be 1D.");
          goto fail;
        }
    }
  DBGPRINTF("Check dimension and size of alms\n");
  for( int i=0; i<nalm; i++ )
    {
      if( (alms[i]->nd != 1) || (alms[i]->descr->type != 'D') )
        {
          PyErr_SetString(PyExc_TypeError,
                      "Type of alms must be complex128 and arrays must be 1D.");
          goto fail;
        }
    }

  /* Now, I check that all alms have the same size and that it is compatible with
     lmax and mmax */
  DBGPRINTF("Check alms have identical size\n");
  int szalm;
  szalm = -1;
  for( int i=0; i<nalm; i++ )
    {
      if( i==0 )
        szalm = alms[i]->dimensions[0];
      else if( alms[i]->dimensions[0] != szalm )
        {
          PyErr_SetString(PyExc_ValueError,
                          "All alms arrays must have same size.");
          goto fail;
        }
    }
  if( szalm != int(Alm< xcomplex<double> >::Num_Alms(lmax,mmax)) )
    {
      PyErr_SetString(PyExc_ValueError,
                      "lmax and mmax are not compatible with size of alms.");
      goto fail;
    }
  DBGPRINTF("Alms have all size %d\n", szalm);

  /* Set the objects Alm */
  DBGPRINTF("Set alm objects\n");
  for( int i=0; i<nalm; i++)
    {
      DBGPRINTF("Setting almalms[%d]\n", i);
      arr< xcomplex<double> > * alm_arr;
      alm_arr = new arr< xcomplex<double> >((xcomplex<double>*)alms[i]->data, szalm);
      DBGPRINTF("Set...\n");
      almalms[i].Set(*alm_arr, lmax, mmax);
      delete alm_arr;
    }


  /* Now, I can loop over l,
     for each l, I make the Cholesky decomposition of the correlation matrix
     given by the cls[*][l]
  */
  DBGPRINTF("Start loop over l\n");
  for( int l=0; l<=lmax; l++ )
    {
      DBGPRINTF("l=%d\n", l);
      /* fill the matrix of cls */
      for( int i=0; i<ncl; i++ )
        {
          if( cls[i] == NULL )
            mat[i] = 0.0;
          else if( cls[i]->dimensions[0] < l )
            mat[i] = 0.0;
          else
            {
              if( cls[i]->descr->type == 'f' )
                mat[i] = (double)(*((float*)PyArray_GETPTR1(cls[i],l)));
              else
                mat[i] = *((double*)PyArray_GETPTR1(cls[i],l));
            }
        }

      /* Make the Cholesky decomposition */
      cholesky(nalm, mat, res);

      if( l == 400 )
        {
          DBGPRINTF("matrice: ");
          for( int i=0; i<ncl; i++ )
            DBGPRINTF("%d: %lg  ", i, mat[i]);
          DBGPRINTF("\n");
          DBGPRINTF("cholesky: ");
          for( int i=0; i<ncl; i++ )
            DBGPRINTF("%d: %lg  ", i, res[i]);
          DBGPRINTF("\n");
        }

      /* Apply the matrix to each m */

      /* m=0 */
      DBGPRINTF("   m=%d: ", 0);
      for( int i=nalm-1; i>=0; i-- )
        {
          double x;
          x = 0.0;
          almalms[i](l,0)=xcomplex<double>(almalms[i](l,0).real(),0.0);
          for( int j=0; j<=i; j++ )
            x += res[getidx(nalm,i,j)]*almalms[j](l,0).real();
          almalms[i](l,0)=xcomplex<double>(x,0.);
          DBGPRINTF(" %lg %lg ;", almalms[i](l,0).real(), almalms[i](l,0).imag());
        }
      DBGPRINTF("\n");

      /* m > 1 */
      for( int m=1; m<=l; m++ )
        {
          DBGPRINTF("   m=%d: ", m);
          for( int i=nalm-1; i>=0; i-- )
            {
              double xr, xi;
              xi = xr = 0.0;
              for( int j=0; j<=i; j++ )
                {
                  xr += res[getidx(nalm,i,j)]*almalms[j](l,m).real();
                  xi += res[getidx(nalm,i,j)]*almalms[j](l,m).imag();
                  DBGPRINTF("(res[%d]=%lg, alm=%lg,%lg) %lg %lg", (int)getidx(nalm,i,j),
                            res[getidx(nalm,i,j)],
                            almalms[j](l,m).real(), almalms[j](l,m).imag(),
                            xr, xi);
                }
              almalms[i](l,m)=xcomplex<double>(xr/sqrt_two,xi/sqrt_two);
              DBGPRINTF(" xre,xim[%d]: %lg %lg ;", i,
                        almalms[i](l,m).real(), almalms[i](l,m).imag());
            }
          DBGPRINTF("\n");
      }
   }

  /* Should be finished now... */
  XFREE(cls);
  XFREE(alms);
  XDELETE(almalms);
  XFREE(mat);
  XFREE(res);
  Py_INCREF(Py_None);
  return Py_None;

  /* To be done before returning */
 fail:
  XFREE(cls);
  XFREE(alms);
  XDELETE(almalms);
  XFREE(mat);
  XFREE(res);

  return NULL;
}

PyObject *
_getn(PyObject *self, PyObject *args) {
  long s;
  if( !PyArg_ParseTuple(args, "l", &s) ) {
  	std::cout << "This function takes an integer as argument."
			  << std::endl;
 	 return Py_BuildValue("l",0);

  }
  long n = getn(s);
  return Py_BuildValue("l",n);
}

static PyMethodDef methods[] = {
  {"_synalm", (PyCFunction)_synalm, METH_VARARGS | METH_KEYWORDS,
   "Compute alm's given cl's and unit variance random arrays.\n"},
  {"_getn", _getn, METH_VARARGS,
   "Compute number n such that n(n+1)/2 is equal to the argument.\n"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "synalm",
  NULL, -1, methods
};

PyMODINIT_FUNC
PyInit_synalm(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
