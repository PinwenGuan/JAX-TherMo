from diffthermo import *
import numpy as np
import time

params_record=np.loadtxt('params_record')
predicted_phase_boundary=[]
Tmin=800
Tmax=2300
dT=10
T_sample=[Tmin+i*dT for i in range(int((Tmax-Tmin)/dT+1))]
for T in T_sample:
#for T in [1000,1400]:
    predicted_phase_boundary.append([T,phase_boundary(params_record[-1],T=T,tolerance=1e-9)])

points12=[]
points11=[]
for i in range(len(predicted_phase_boundary)):
    opb=ordered_phase_boundary(predicted_phase_boundary[i][1])
    for j in range(len(opb[0])):
        if abs(np.array(opb[0][j])-np.array([1,2])).max()==0:
            points12.append(np.concatenate((np.array([predicted_phase_boundary[i][0]]),opb[1][j][0])))
        if abs(np.array(opb[0][j])-np.array([1,1])).max()==0:
            points11.append(np.concatenate((np.array([predicted_phase_boundary[i][0]]),opb[1][j][0])))

points12=np.array(points12)
points11=np.array(points11)
np.savetxt('predicted_phase_boundary_12',points12)
np.savetxt('predicted_phase_boundary_11',points11)
