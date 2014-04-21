#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, int, linspace, any, all
from numpy.random import random as random
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from numpy.random import normal as normal

from collections import deque

import gtk, gobject

NMAX = 2*1e7 # maxmimum number of nodes
N = 800 # image resolution
ZONES = N/100 # number of zones on each axis
ONE = 1./N # pixelsize

BACK = [0.1]*3
FRONT = [0.8]*3
CONTRASTA = [0.84,0.37,0] # orange
CONTRASTB = [0.53,0.53,1] # lightblue
CONTRASTC = [0.84,1,0]

X_MIN = 0+10*ONE # border
Y_MIN = 0+10*ONE #
X_MAX = 1-10*ONE #
Y_MAX = 1-10*ONE #

RAD = 40*ONE # 
RAD_SCALE = 0.9
R_RAND_SIZE = 7 
CK_MAX = 7 # max number of allowed branch attempts from a node

UPDATE_NUM = 1 # write image this often
#TIMEOUT = 100 ## 5 is quite fast
TIMEOUT = 5 ## 5 is quite fast

SEARCH_ANGLE_MAX = pi
SEARCH_ANGLE_EXP = 0.15
SOURCE_NUM = 1

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

    window.connect("destroy", self.__write_image_and_exit)
    darea = gtk.DrawingArea()
    darea.connect("expose-event", self.expose)
    window.add(darea)
    window.show_all()

    self.darea = darea

    self.num_img = 0
    self.miss = 0
    self.itt = 0

    #gobject.idle_add(self.step_wrap)
    gobject.timeout_add(TIMEOUT,self.step_wrap)
    gtk.main()

  def __write_image_and_exit(self,*args):

    self.sur.write_to_png('on_exit.png')
    gtk.main_quit(*args)

  def __init_data(self):

    self.Z = [[] for i in xrange((ZONES+2)**2)]

    self.R = np.zeros(NMAX,'float') # radius
    self.X = np.zeros(NMAX,'float') # x position
    self.Y = np.zeros(NMAX,'float') # y position
    self.THE = np.zeros(NMAX,'float') # angle
    self.GE = np.zeros(NMAX,'float') # generation
    self.P = np.zeros(NMAX,'int') # index of parent node
    self.C = np.zeros(NMAX,'int') # number of branch attempts
    self.D = np.zeros(NMAX,'int')-1 # index of first descendant
    self.B = np.zeros(NMAX,'int') # index of branch
    self.DQ = deque()

    self.num = 0

    ## only one source node
    i = 0

    x = 0.5
    y = 0.5

    self.X[i] = x
    self.Y[i] = y
    self.THE[i] = random()*pi*2.
    self.GE[i] = 1
    self.P[i] = -1 # no parent
    self.R[i] = RAD
    self.B[i] = i
    self.DQ.append(i)

    z = get_z(x,y)
    self.Z[z].append(self.num)
    self.num += 1

    ## draw inicator circle
    self.ctx.set_source_rgba(*CONTRASTB)
    self.circle(x,y,RAD*0.5)

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)
    ctx.set_source_rgb(*BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx
    self.ctx.set_line_width(ONE)

  def line(self,x1,y1,x2,y2):

    #self.ctx.set_line_width(ONE*2.)

    self.ctx.set_source_rgba(*FRONT)
    self.ctx.move_to(x1,y1)
    self.ctx.line_to(x2,y2)
    self.ctx.stroke()

  def circle(self,x,y,r):

    self.ctx.arc(x,y,r,0,pi*2.)
    self.ctx.fill()

  def circle_stroke(self,x,y,r):

    self.ctx.arc(x,y,r,0,pi*2.)
    self.ctx.stroke()

  def circles(self,x1,y1,x2,y2,r):

    dx = x1-x2
    dy = y1-y2
    dd = sqrt(dx*dx+dy*dy)

    n = int(dd/ONE)
    n = n if n>6 else 6

    a = arctan2(dy,dx)

    #scale = random(n)*dd
    scale = linspace(0,dd,n)

    xp = x1-scale*cos(a)
    yp = y1-scale*sin(a)

    for x,y in zip(xp,yp):
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

    print self.num

    return res

  def step(self):

    self.itt += 1
    num = self.num

    try:

      k = self.DQ.pop()

    except IndexError:

      ## no more live nodes.
      return False, False

    self.C[k] += 1

    if self.C[k]>CK_MAX:

      ## node is dead

      ## this is inefficient. but it does not matter for small canvases
      self.ctx.set_source_rgb(*CONTRASTC)
      self.circle(self.X[k],self.Y[k],ONE*4)

      return True, False

    #r = RAD + random()*ONE*R_RAND_SIZE
    r = self.R[k]*RAD_SCALE if self.D[k]>-1 else self.R[k]
    b = num if self.D[k]>-1 else self.B[k]

    if r<ONE:

      ## node dies

      self.ctx.set_source_rgb(*CONTRASTC)
      self.circle(self.X[k],self.Y[k],ONE*4)

      self.C[k] = CK_MAX+1
      return True, False

    ge = self.GE[k]+1 if self.D[k]>-1 else self.GE[k]

    angle = normal()*SEARCH_ANGLE_MAX
    the = self.THE[k] + (1.-1./((ge+1)**SEARCH_ANGLE_EXP))*angle

    x = self.X[k] + sin(the)*r
    y = self.Y[k] + cos(the)*r

    ## stop nodes at edge of canvas
    if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:

      ## node is outside canvas
      return True, False

    ## stop nodes at edge of circle
    ## remember to set initial node inside circle.
    #circle_rad = sqrt(square(x-0.5)+square(y-0.5))
    #if circle_rad>CIRCLE_RADIUS:

      ### node is outside circle
      #return True,False
    
    try:

      inds = near_zone_inds(x,y,self.Z,k)
    except IndexError:

      ## node is outside zonemapped area
      self.DQ.appendleft(k)
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
      self.GE[num] = ge
      self.B[num] = b

      ## set first descendant if node has no descendants
      if self.D[k]<0:
        self.D[k] = num

      z = get_z(x,y) 

      self.Z[z].append(num)

      self.ctx.set_line_width(ONE*2)
      self.ctx.set_source_rgb(*FRONT)
      self.line(self.X[k],self.Y[k],x,y)

      self.ctx.set_source_rgb(*CONTRASTB)
      self.circle(x,y,ONE*4)

      #self.ctx.set_line_width(ONE)
      #self.ctx.set_source_rgb(*CONTRASTA)
      #self.circle_stroke(x,y,r*0.5)

      #self.ctx.set_source_rgb(*FRONT)
      #self.circles(self.X[k],self.Y[k],x,y,r*0.3)

      #self.ctx.set_source_rgb(*CONTRASTB)
      #self.circle_stroke(x,y,r*0.5)

      self.DQ.appendleft(num)
      self.DQ.appendleft(k)

      self.num += 1

      ## node was added
      return True, True

    if not good and len(inds)>0:

      ## can we merge two branches?

      if mask.sum()>1:

        ms = np.argsort(dd)
        mks = ms[:2]
        mk = inds[mks[0]]

        if 2*dd[mks[0]]<dd[mks[1]] and self.P[mk]!=k and self.C[mk]<CK_MAX:

          self.ctx.set_source_rgb(*CONTRASTA)
          #self.circles(self.X[k],self.Y[k],self.X[mk],self.Y[mk],r*0.3)

          self.ctx.set_line_width(ONE*2)
          self.ctx.set_source_rgb(*FRONT)
          self.line(self.X[k],self.Y[k],self.X[mk],self.Y[mk])

          self.ctx.set_source_rgb(1,0,0)
          self.circle(self.X[mk],self.Y[mk],ONE*5)

          #self.C[k] = CK_MAX+1
          self.C[mk] = CK_MAX+1

    ## failed to place node
    self.DQ.appendleft(k)
    return True, False


def main():

  render = Render(N)

if __name__ == '__main__':
  main()

