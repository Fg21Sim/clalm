#!/usr/bin/env python3
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
ext_modules = [
	Extension(
		'synalm', 
		sources = [
			'synalm.cc', 
			'alm.cc', 
			'error_handling.cc',
		],
		language='c++',
		extra_compile_args = ['-std=c++11'],
	)
] 
ext_modules.append(cythonize(['alm2cl.pyx', ], language_level='3')[0])
setup(name='clalm', 
		version='0.0', 
		include_dirs=[numpy.get_include()],
		ext_modules= ext_modules,
		py_modules = ['cl2alm'],
)
