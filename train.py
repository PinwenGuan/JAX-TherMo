from diffthermo import *
import numpy as np
import time

#target_phase_boundary has the form:
#[(T1,x1),...,(Tn,xn)]
from jax.nn import relu
l_record=[]
l_phase_boundary_record=[]
l_driving_force_record=[]
l_immiscibility_record=[]
def loss(params_opt,target_phase_boundary):
    R=8.314
    loss_phase_boundary=[]; loss_driving_force=[]; loss_immiscibility=[]
    for i in range(len(target_phase_boundary)):
        T,target_x0=target_phase_boundary[i]
        a1=params_opt[0]*1000;b1=params_opt[1]*1000;
        g11=0;g21=0
        a2=params_opt[2]*1000;b2=params_opt[3]*1000;
        g12=13054-9.623*T;g22=18753-8.372*T 
        params1=(a1,b1,g11,g21)
        params2=(a2,b2,g12,g22)
        x0=phase_boundary(params_opt,T)
        target_phase_type,target_x=ordered_phase_boundary(target_x0)
        phase_type,x=ordered_phase_boundary(x0)
        for j in range(len(target_phase_type)):
            phase_type_in_prediction=False
            for k in range(len(phase_type)):
                phase_difference=abs(jnp.array(target_phase_type[j])-jnp.array(phase_type[k]))
                if phase_difference.max()==0:
                    phase_type_in_prediction=True
                    loss_phase_boundary.append(x[k]-target_x[j])
            if not phase_type_in_prediction:
                if target_phase_type[j][0]!=target_phase_type[j][1]:
                    print('Target phases '+str(target_phase_type[j])+' at T='+str(T)+
                    ' not in prediction: using driving force mode!')
                    loss_driving_force.append(df_min(params_opt,T,n_df_min=30))
                else:
                    print('Target phases '+str(target_phase_type[j])+' at T='+str(T)+
                    ' not in prediction: using phase separation mode!')
                    if target_phase_type[j][0]==1:
                        params_spinodal=params1
                    else:
                        params_spinodal=params2                  
                    hess_min=Hessian_min(G,T,*params_spinodal)/(R*T)
                    loss_immiscibility.append(relu(hess_min))                    
    l_phase_boundary=0;l_driving_force=0; l_immiscibility=0
    if len(loss_phase_boundary)!=0:
        loss_phase_boundary=jnp.concatenate(tuple(loss_phase_boundary))
        l_phase_boundary=jnp.mean(loss_phase_boundary**2)
    if len(loss_driving_force)!=0:
        loss_driving_force=jnp.array(loss_driving_force)
        l_driving_force=jnp.mean(loss_driving_force**2)
    if len(loss_immiscibility)!=0:
        loss_immiscibility=jnp.array(loss_immiscibility)
        l_immiscibility=jnp.mean(loss_immiscibility**2)
    alpha=100
    l=alpha*l_phase_boundary+l_driving_force+l_immiscibility
    l_phase_boundary_record.append(alpha*l_phase_boundary)
    l_driving_force_record.append(l_driving_force)
    l_immiscibility_record.append(l_immiscibility)
    l_record.append(l)
    return l

# generate target_phase_boundary
target_value=jnp.array([14609.0,11051.0,8414.0,19799.0])/1000
target_phase_boundary=[]
for T in [1000,1200,1400,1600,1800,2000,2200]:
#for T in [1000,1400,1800,2200]:
    target_phase_boundary.append([T,phase_boundary(target_value,T=T)])

grad_1=jax.value_and_grad(loss)
n_df_min=100
params_record=[]
g_record=[]
def np_grad_1(params_opt):
    start_time = time.time()
    print(params_opt)
    f,g=grad_1(params_opt,target_phase_boundary)
    params_record.append(params_opt)
    g_record.append(g)
    step_time = time.time() - start_time
    print(f'Step time: {step_time} sec')
    return f,np.array(g)

x, y, info = scipy.optimize.fmin_l_bfgs_b(np_grad_1,x0=target_value+9*jnp.array([-1.0,-1.0,1.0,-1.0]),args=(),bounds=[(-25,25),(-25,25),(-25,25),(-25,25)],maxfun=n_df_min)

#---save

l_record=np.array([data_converter(i) for i in l_record])
np.savetxt('l_record',l_record)

l_phase_boundary_record=np.array([data_converter(i) for i in l_phase_boundary_record])
np.savetxt('l_phase_boundary_record',l_phase_boundary_record)

l_driving_force_record=np.array([data_converter(i) for i in l_driving_force_record])
np.savetxt('l_driving_force_record',l_driving_force_record)

l_immiscibility_record=np.array([data_converter(i) for i in l_immiscibility_record])
np.savetxt('l_immiscibility_record',l_immiscibility_record)

params_record=np.array(params_record)
np.savetxt('params_record',params_record)

g_record=np.array(g_record)
np.savetxt('g_record',g_record)
