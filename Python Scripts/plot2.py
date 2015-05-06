from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X = [0,10,10,10,10, 20, 20, 20, 20]
Y = [0,1,2,3,10,1,2,3,10]
ZPython = [0,9.133,16.940,25.551,25.551,22.179,45.177,83.685,264.194]
ZCuda = [0,0.00586,0.01174,0.01753,0.05816,0.011,0.02194,0.03284,0.10922]

ZPythonLarge = [0,43.508,78.817,121.933,400.653,66.939,135.665,186.916,625.291 ]
ZCudaLarge = [0,0.03056,0.06102,0.09125,0.30525,0.05364,0.107,0.161,0.536]

ax.scatter(X,Y,ZPythonLarge, c= 'b', marker = '*')
ax.scatter(X,Y,ZCudaLarge, c= 'r', marker = 'o')

ax.set_xlabel('x-#Clusters')
ax.set_ylabel('y-#Iterations')
ax.set_zlabel('z-Time(seconds)')



plt.show()

