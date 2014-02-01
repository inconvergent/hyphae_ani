#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt, square, int, linspace, any, all
from numpy.random import random as rand
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from numpy.random import normal as norm

import gtk, gobject

NMAX = 2*1e7 # maxmimum number of nodes
N = 1080 # image resolution
ZONES = N/10 # number of zones on each axis
ONE = 1./N # pixelsize
BACK = 1.
FRONT = 0.
X_MIN = 0+10*ONE # border
Y_MIN = 0+10*ONE #
X_MAX = 1-10*ONE #
Y_MAX = 1-10*ONE #

MISS_MAX = 1000 # restart on MISS_MAX failed branch attempts

RAD = 10*ONE # 
RAD_SCALE = 0.8
R_RAND_SIZE = 7 
CK_MAX = 30 # max number of allowed branch attempts from a node

UPDATE_NUM = 80 # write image this often

SEARCH_ANGLE = 0.22*pi
SOURCE_NUM = 20

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


class Render(object):

  def __init__(self,n):

    self.n = n

    self.__init_cairo()
    self.__init_data()

    window = gtk.Window()
    window.resize(self.n, self.n)
    window.connect("destroy", gtk.main_quit)
    darea = gtk.DrawingArea()
    darea.connect("expose-event", self.expose)
    window.add(darea)
    window.show_all()

    self.darea = darea

    self.num_img = 0
    self.miss = 0
    self.itt = 0

    gobject.idle_add(self.step_wrap)
    gtk.main()

  def __init_data(self):

    self.Z = [[] for i in xrange((ZONES+2)**2)]

    self.P = np.zeros(NMAX,'int') # index of parent node
    self.D = np.zeros(NMAX,'int')-1 # index of first descendant
    self.R = np.zeros(NMAX,'float') # radius
    self.X = np.zeros(NMAX,'float') # x position
    self.Y = np.zeros(NMAX,'float') # y position
    self.THE = np.zeros(NMAX,'float') # angle
    self.C = np.zeros(NMAX,'int') # number of branch attempts

    self.num = 0

    for i in xrange(SOURCE_NUM):

      ## randomly on canvas
      x = X_MIN + rand()*(X_MAX-X_MIN) 
      y = Y_MIN + rand()*(Y_MAX-Y_MIN) 

      ## on circle
      #x = 0.5 + sin((i*pi*2)/float(SOURCE_NUM-1))*0.3
      #y = 0.5 + cos((i*pi*2)/float(SOURCE_NUM-1))*0.3

      self.X[i] = x
      self.Y[i] = y
      self.THE[i] = rand()*pi*2.
      self.P[i] = -1 # no parent
      self.R[i] = RAD

      z = get_z(x,y)
      self.Z[z].append(self.num)
      self.num += 1

      ## draw inicator circle
      self.ctx.set_source_rgba(1,0,0,0.4)
      self.circle(x,y,RAD*0.5)

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

  def circle(self,x,y,r):

    self.ctx.arc(x,y,r,0,pi*2.)
    self.ctx.fill()

  def expose(self,*args):

    cr = self.darea.window.cairo_create()
    cr.set_source_surface(self.sur,0,0)
    cr.paint()

  def step_wrap(self,*args):

    res, added_new = self.step()

    if not self.num%UPDATE_NUM and added_new:
      self.expose()
      self.miss = 0
      fn = 'image{:05d}.png'.format(self.num_img)
      #self.sur.write_to_png(fn)
      self.num_img += 1
      print fn, self.num

    ## if self.miss is too large the animation will restart
    #if not added_new:
      #self.miss += 1

    #if self.miss>MISS_MAX:

      #self.__init_cairo()
      #self.__init_data()
      #self.miss = 0
      #return True

    return res

  def step(self):

    self.itt += 1
    num = self.num

    k = int(rand()*num)
    self.C[k] += 1

    if self.C[k] > CK_MAX:
      return True, False

    #r = RAD + rand()*ONE*R_RAND_SIZE
    r = self.R[k]*RAD_SCALE if self.D[k]>-1 else self.R[k]

    #sa = norm()*SEARCH_ANGLE
    sa = norm()*(1.-r/(RAD+ONE))*pi
    the = sa+self.THE[k]

    x = self.X[k] + sin(the)*r
    y = self.Y[k] + cos(the)*r

    if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:
      return True, False
    
    try:
      inds = near_zone_inds(x,y,self.Z,k)
    except IndexError:
      return True, False

    good = True
    if len(inds)>0:
      dd = square(self.X[inds]-x) + square(self.Y[inds]-y)

      sqrt(dd,dd)
      mask = dd*2 >= self.R[inds]+r
      good = mask.all()
      
    if good: 
      self.X[num] = x
      self.Y[num] = y
      self.R[num] = r
      self.THE[num] = the
      self.P[num] = k

      ## set first descendant if node has no descendants
      if self.D[k]<0:
        self.D[k] = num

      z = get_z(x,y) 

      self.Z[z].append(num)

      self.ctx.set_source_rgba(0,0,0,0.9)
      self.line(self.X[k],self.Y[k],x,y)
      #self.circle(x,y,r*0.5)

      self.num += 1

      return True, True

    return True, False


def main():

  render = Render(N)

if __name__ == '__main__':
  main()

