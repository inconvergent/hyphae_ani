#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt, square, int, linspace
from numpy.random import random as rand
import numpy as np
from time import time as time
from operator import itemgetter
from numpy.random import normal as norm

import gtk
import gobject


any = np.any
all = np.all

N = 1000
ZONES = N/20
BACK = 1.
FRONT = 0.

NMAX = 2*1e7

RAD = 3;
R_RAND_SIZE = 10
CK_MAX = 20

SEARCH_ANGLE = 0.3*pi
SOURCE_NUM = 3


class Render(object):

  def __init__(self,n):

    self.N = n

    self.__data_init()

    window = gtk.Window()
    window.resize(n, n)
    window.connect("destroy", gtk.main_quit)
    darea = gtk.DrawingArea()
    darea.connect("expose-event",self.expose)
    window.add(darea)
    window.show_all()

    self.window = window
    self.darea = darea
    self.cr = darea.window.cairo_create()

    gtk.main()

  def __data_init(self):

    self.Z = [[] for i in xrange((ZONES+2)**2)]

    self.R = np.zeros(NMAX,dtype=np.float)
    self.X = np.zeros(NMAX,dtype=np.float)
    self.Y = np.zeros(NMAX,dtype=np.float)
    self.THE = np.zeros(NMAX,dtype=np.float)

    self.C = np.zeros(NMAX,dtype=np.int)

    self.num = 0

    for i in xrange(SOURCE_NUM):

      self.X[i] = rand()*self.N
      self.Y[i] = rand()*self.N
      self.THE[i] = rand()*pi*2.

      z = get_z(self.X[i],self.Y[i])
      self.Z[z].append(self.num)
      self.num += 1

  def expose(self,*args):

    X = self.X
    Y = self.Y
    Z = self.Z
    C = self.C
    THE = self.THE
    R = self.R

    num = self.num

    itt = 0
    t = time()

    while True:
      k = int(rand()*num)
      C[k] += 1

      if C[k] > CK_MAX:
        continue

      the = get_relative_search_angle()+THE[k]
      r = RAD  + rand()*R_RAND_SIZE
      x = X[k] + sin(the)*r
      y = Y[k] + cos(the)*r
      
      try:
        inds = near_zone_inds(x,y,Z,k)
      except IndexError:
        continue

      good = True
      if len(inds)>0:
        dd = square(X[inds]-x) + square(Y[inds]-y)

        sqrt(dd,dd)
        mask = dd*2 >= (R[inds] + r)
        good = mask.all()
        
      if good: 
        X[num] = x
        Y[num] = y
        R[num] = r
        THE[num] = the

        z = get_z(x,y) 

        Z[z].append(num)
        
        self.line(X[k],Y[k],x,y)

        num += 1
      
      itt += 1
      if not itt%100:
        print num, time()-t
        t = time()

  def line(self,x1,y1,x2,y2):

    self.cr.set_source_rgba(FRONT,FRONT,FRONT)
    self.cr.set_line_width(2.)
    self.cr.move_to(x1,y1)
    self.cr.line_to(x2,y2)
    self.cr.stroke()


def near_zone_inds(x,y,Z,k):
  
  i = 1+int(x/ZONES) 
  j = 1+int(y/ZONES) 
  ij = np.array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*ZONES+\
       np.array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])

  it = itemgetter(*ij)
  its = it(Z)
  inds = np.array([b for a in its for b in a if not b==k])

  return inds

def get_z(x,y):

  i = 1+int(x/ZONES) 
  j = 1+int(y/ZONES) 
  z = i*ZONES+j
  return z

def get_relative_search_angle():

  a = norm()*SEARCH_ANGLE
  #a = (0.5-rand())*SEARCH_ANGLE
  
  return a

def main():

  render = Render(N)

if __name__ == '__main__':
  main()

