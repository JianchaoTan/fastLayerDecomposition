from numpy import *
import json

ts = linspace(0,1,1000)
colors = asarray([ sin(6*pi*ts)*.5+.5, cos(6*pi*ts)*.5+.5, ts ]).T
json.dump({'float_colors': colors.tolist()}, open("color_spiral.js",'wb'))
