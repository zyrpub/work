#!/usr/bin/env python

import os, sys
import subprocess
import random, time
import inspect
import collections
import math
# from shutil import copy2, move as movefile
# mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# sys.path.append(mypydir)
# sys.path.append(mypydir+"/mytools")

import numpy as np
import scipy.linalg as la
# import numpy.linalg as la


def csp(*tasks):
	if len(tasks) < 2:
		print "Must have at least 2 tasks for filtering."
		return (None,) * len(tasks)
	else:
		filters = ()
		# CSP algorithm
		# For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
		iterator = range(0,len(tasks))
		for x in iterator:
			# Find Rx
			Rx = covarianceMatrix(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Rx += covarianceMatrix(tasks[x][t])
			Rx = Rx / len(tasks[x])

			# Find not_Rx
			count = 0
			not_Rx = Rx * 0
			for not_x in [element for element in iterator if element != x]:
				for t in range(0,len(tasks[not_x])):
					not_Rx += covarianceMatrix(tasks[not_x][t])
					count += 1
			not_Rx = not_Rx / count

			# Find the spatial filter SFx
			SFx = spatialFilter(Rx,not_Rx)
			filters += (SFx,)

			# Special case: only two tasks, no need to compute any more mean variances
			if len(tasks) == 2:
				filters += (spatialFilter(not_Rx,Rx),)
				break
		return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)

	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	return SFa.astype(np.float32)



# --------------------=-=-=-=-=-=-==-=-=-=-============

def whitener(C, rtol=1e-15):
  '''
  Calculate the whitening transform for signals with covariance C.
  The whitening transform is used to remove covariance between
  signals, and can be regarded as principal component analysis with
  rescaling and an optional rotation. The whitening transform is
  calculated as C^{-1/2}, and implemented to work with rank-deficient
  covariance matrices.
  Parameters
  ----------
  C : array-like, shape (p, p)
	Covariance matrix of the signals to be whitened.
  rtol : float, optional
	Cut-off value specified as the fraction of the largest eigenvalue
	of C.
  Returns
  -------
  W : array, shape (p, p)
	Symmetric matrix with spatial filters in the rows.
  See also
  --------
  car : common average reference
  Examples
  --------
  >>> X = np.random.randn(3, 100) 
  >>> W = whitener(np.cov(X))
  >>> X_2 = np.dot(W, X - np.mean(X, axis=1).reshape(-1, 1))
  The covariance of X_2 is now close to the identity matrix:
  >>> np.linalg.norm(np.cov(X_2) - np.eye(3)) < 1e-10
  True
  '''
  e, E = np.linalg.eigh(C)
  return reduce(np.dot, 
	[E, np.diag(np.where(e > np.max(e) * rtol, e, np.inf)**-.5), E.T])


def outer_n(n):
  '''
  Return a list with indices from both ends. Used for CSP.
  --------
  >>> outer_n(6)
  array([ 0,  1,  2, -3, -2, -1])
  '''
  return np.roll(np.arange(n) - n/2, (n + 1) / 2)


def csp_base(C_a, C_b):
  '''
  Calculate the full common spatial patterns (CSP) transform. 
  The CSP transform finds spatial filters that maximize the variance
  in one condition, and minimize the signal's variance in the other.
  See [1]. Usually, only a subset of the spatial filters is used.
  Parameters
  ----------
  C_a : array-like of shape (n, n)
	Sensor covariance in condition 1.
  C_b : array-like of shape (n, n)
	Sensor covariance in condition 2.
  Returns
  -------
  W : array of shape (m, n)
	A matrix with m spatial filters with decreasing variance in the
	first condition. The rank of (C_a + C_b) determines the number
	of filters m.
  See also
  --------
  In condition 1 the signals are positively correlated, in condition 2
  they are negatively correlated. Their variance stays the same:
  >>> C_1 = np.ones((2, 2))
  >>> C_1
  array([[ 1.,  1.],
		 [ 1.,  1.]])
  >>> C_2 = 2 * np.eye(2) - np.ones((2, 2))
  >>> C_2
  array([[ 1., -1.],
		 [-1.,  1.]])
  The most differentiating projection is found with the CSP transform:
  >>> csp_base(C_1, C_2).round(2)
  array([[-0.5,  0.5],
		 [ 0.5,  0.5]])
  '''
  P = whitener(C_a + C_b)
  P_C_b = reduce(np.dot, [P, C_b, P.T])
  _, _, B = np.linalg.svd((P_C_b))
  return np.dot(B, P.T)


def csp2(C_a, C_b, m):
  '''
  Calculate common spatial patterns (CSP) transform. 
  The CSP transform finds spatial filters that maximize the variance
  in one condition, and minimize the signal's variance in the other.
  See [1]. Usually, only a subset of the spatial filters is used.
  Parameters
  ----------
  C_a : array-like of shape (n, n)
	Sensor covariance in condition 1.
  C_b : array-like of shape (n, n)
	Sensor covariance in condition 2.
  m : int
	The number of CSP filters to extract.
  Returns
  -------
  W : array of shape (m, n)
	A matrix with m/2 spatial filters that maximize the variance in
	one condition, and m/2 that maximize the variance in the other.
  See also
  --------
  csp_base : full common spatial patterns transform.
  outer_n : pick indices from both sides.
 
  We construct two nearly identical covariance matrices for condition
  1 and 2:
  >>> C_1 = np.eye(4)
  >>> C_2 = np.eye(4)
  >>> C_2[1, 3] = 1
  The difference between the conditions is in the 2nd and 4th sensor:
  >>> C_2 - C_1
  array([[ 0.,  0.,  0.,  0.],
		 [ 0.,  0.,  0.,  1.],
		 [ 0.,  0.,  0.,  0.],
		 [ 0.,  0.,  0.,  0.]])
  The two most differentiating projections are found with the CSP transform.
  Indeed, it projects the same sensors:
  >>> csp(C_1, C_2, 2).round(2)
  array([[ 0.  ,  0.37,  0.  ,  0.6 ],
		 [ 0.  , -0.6 ,  0.  ,  0.37]])
  '''
  W = csp_base(C_a, C_b)
  assert W.shape[1] >= m
  return W[outer_n(m)]

if __name__ == "__main__":
	C_1 = np.eye(4)
	C_2 = np.eye(4)
	C_2[1, 3] = 1
	sf2=csp2(C_1, C_2, 4)
	print(sf2)
	sf1=spatialFilter(C_1, C_2)
	print(sf1)