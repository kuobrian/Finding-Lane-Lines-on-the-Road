# -*- coding: utf-8 -*-

__author__ = 'user'

import numpy

import numpy as np                                                           

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore', np.RankWarning)  # 可排除錯誤的訊息

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0,6.0,7.0,8.0,9.0,10.0])

y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0,-0.8,-0.7,-0.2,0.2,0.6])

a3 = np.polyfit(x, y, 3) ## 3階乘，即x最高次方係数是3

print "a3", a3                #印出3階乘的各個係數

a10 = np.polyfit(x, y, 10) ## 10階乘，即x最高次方係数是10

print "a10", a10

p3 = np.poly1d(a3)      #產生出線性方程式

print "p3",p3                #印出3階乘的各值

p10 = np.poly1d(a10)

xp=np.linspace(0,10,20)


for i in range(len(xp)):

  print "xp=%.6f,Y=%.6f" % (xp[i],p3(xp)[i])

#warnings.simplefilter('ignore', np.RankWarning)

plt.plot(x, y, ".", markersize = 10) # 點為原始數據

plt.plot(xp, p3(xp), "r--") # 紅線是a3的3階乘

plt.plot(xp, p10(xp), "b--") # 藍線是a10的10階乘
plt.show()
