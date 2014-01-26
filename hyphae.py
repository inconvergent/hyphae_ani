#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt, square, int, linspace
from numpy.random import random as rand
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from numpy.random import normal as norm

import gtk, gobject

any = np.any
all = np.all

N = 800
ZONES = N/20
ONE = 1./N
BACK = 1.
FRONT = 0.
X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

MAX_NUM = 10000

RAD = 3*ONE;
R_RAND_SIZE = 5
CK_MAX = 30

UPDATE_NUM = 200

LINE_NOISE = 1.
SEARCH_ANGLE = 0.1*pi
SOURCE_NUM = 3

ALPHA = 0.5
GRAINS = 3


def near_zone_inds(x,y,Z,k):
  
  i = 1+int(x*ZONES) 
  j = 1+int(y*ZONES) 
  ij = np.array([i-1,i,i+1,i-1,i,i+1,i-1,i,i+1])*ZONES+\
       np.array([j+1,j+1,j+1,j,j,j,j-1,j-1,j-1])

  it = itemgetter(*ij)
  its = it(Z)
  inds = np.array([b for a in its for b in a if not b==k])

  return inds

def get_z(x,y):

  i = 1+int(x*ZONES) 
  j = 1+int(y*ZONES) 
  z = i*ZONES+j
  return z

def get_relative_search_angle():

  a = norm()*SEARCH_ANGLE
  #a = (0.5-rand())*SEARCH_ANGLE
  
  return a

class Render(object):

  def __init__(self,n):

    self.n = n

    self.__init_data()
    self.__init_cairo()

    window = gtk.Window()
    window.resize(self.n, self.n)
    window.connect("destroy", gtk.main_quit)
    darea = gtk.DrawingArea()
    darea.connect("expose-event", self.expose)
    window.add(darea)
    window.show_all()

    self.darea = darea

    gobject.idle_add(self.step_wrap)
    gtk.main()

  def __init_data(self):

    Z = [[] for i in xrange((ZONES+2)**2)]

    nmax = 2*1e7
    R = np.zeros(nmax,dtype=np.float)
    X = np.zeros(nmax,dtype=np.float)
    Y = np.zeros(nmax,dtype=np.float)
    THE = np.zeros(nmax,dtype=np.float)

    C = np.zeros(nmax,dtype=np.int)

    num = 0

    for i in xrange(SOURCE_NUM):

      X[i] = X_MIN + rand()*(X_MAX-X_MIN) 
      Y[i] = Y_MIN + rand()*(Y_MAX-Y_MIN) 
      THE[i] = rand()*pi*2.

      z = get_z(X[i],Y[i])
      Z[z].append(num)
      num += 1

    self.X = X
    self.Y = Y
    self.R = R
    self.Z = Z
    self.C = C
    self.THE = THE
    self.num = num

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def line(self,x1,y1,x2,y2):

    self.ctx.set_source_rgba(FRONT,FRONT,FRONT)
    self.ctx.set_line_width(ONE*2.)
    self.ctx.move_to(x1,y1)
    self.ctx.line_to(x2,y2)
    self.ctx.stroke()

  def expose(self,*args):

    cr = self.darea.window.cairo_create()
    cr.set_source_surface(self.sur,0,0)
    cr.paint()

  def step_wrap(self,*args):

    res, added_new = self.step()

    if not self.num%UPDATE_NUM and added_new:
      self.expose()

    if self.num>MAX_NUM:

      self.__init_data()
      self.__init_cairo()

    return res

  def step(self):

    X = self.X
    Y = self.Y
    R = self.R
    Z = self.Z
    C = self.C
    THE = self.THE
    num = self.num

    k = int(rand()*num)
    C[k] += 1

    if C[k] > CK_MAX:
      return True, False

    the = get_relative_search_angle()+THE[k]
    r = RAD  + rand()*ONE*R_RAND_SIZE
    x = X[k] + sin(the)*r
    y = Y[k] + cos(the)*r

    if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:
      return True, False
    
    try:
      inds = near_zone_inds(x,y,Z,k)
    except IndexError:
      return True, False

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

      self.num += 1

    return True, True


def main():

  render = Render(N)

if __name__ == '__main__':
  main()

