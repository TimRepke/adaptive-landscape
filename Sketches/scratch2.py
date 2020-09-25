# from ctypes import cdll
#
# lib = cdll.LoadLibrary('./externals/LargeVis/Linux/LargeVis.cpython-38-x86_64-linux-gnu.so')
# lv=lib.loaddata()
# largevislib.LargeVis.loaddata([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
import LargeVis
import numpy as np
LargeVis.loadarray(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
#LargeVis.loadfile('../TempData/test.txt')
print('loaded')
LargeVis.run(2)
print('ran')