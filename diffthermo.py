import jax
import jax.numpy as jnp
import numpy as np
import scipy

def qhull(sample,column=[0,1],niter_global=10,tolerance=1e-5):
    column=jnp.array(column)
    link = lambda a,b: jnp.concatenate((a,b[1:]))
    edge = lambda a,b: jnp.concatenate((jnp.array([a]),jnp.array([b])))
    if len(sample) > 2:
        axis = sample[:,column][:,0]
        base = [jnp.take(sample, [jnp.argmin(axis), jnp.argmax(axis)], axis=0)]
        base_working=base.copy()
        for n in range(niter_global):
            base_working_new=base_working.copy()
            for i in range(len(base_working)):
                h = base_working[i][0]; t = base_working[i][1]
                dists = jnp.dot(sample[:,column]-h[column], jnp.dot(jnp.array(((0,-1),(1,0))),(t[column]-h[column])))
                outer = jnp.repeat(sample, dists<-tolerance, axis=0)
                if len(outer):
                    pivot = sample[jnp.argmin(dists)]
                    z=0
                    while (z<=len(base)-1):
                        diff=abs(jnp.array(base[z])-jnp.array(base_working[i])).max()
                        if diff<tolerance:
                            base.pop(z)
                        else:
                            z=z+1
                    base.append([h, pivot])
                    base.append([pivot, t])
                    base_working_new.remove(base_working[i])
                    base_working_new.append([h, pivot])
                    base_working_new.append([pivot, t])
                else:
                    base_working_new.remove(base_working[i])
            base_working=base_working_new
        return base
    else:
        return sample

def G(x,T,*params):
    R=8.314
    a,b,g1,g2=params
    if x==0:
        g=g2
    if x==1:
        g=g1
    else:
        g=x*g2+(1-x)*g1+(a+b*x)*x*(1-x)+R*T*(x*jnp.log(x)+(1-x)*jnp.log(1-x))
    return jnp.reshape(g,())

def Hessian(x,G,T,*params):
    return jax.grad(jax.grad(G))(x,T,*params)

def Hessian_min(G,T,*params):
    x_sample=jnp.concatenate((jnp.linspace(0.1,0.9,9),jnp.array([1e-7,1-1e-7])))
    Hessian_sample=jnp.array([Hessian(i,G,T,*params) for i in x_sample])
    x=x_sample[jnp.argmin(Hessian_sample)]
    hp=jax.grad(Hessian)
    hpp=jax.grad(hp)    
    dx=1
    niter_x=1
    bounds=(1e-7,1.0-1e-7)
    n_df_min=500  
    while (abs(dx)>1e-3) and (niter_x<=n_df_min):
        dx=-hp(x,G,T,*params)/hpp(x,G,T,*params)
        x=x+dx
        if x<bounds[0]:
            x=bounds[0]
        if x>bounds[1]:
            x=bounds[1]
        niter_x=niter_x+1
    H_min=min([Hessian(x,G,T,*params),Hessian(bounds[0],G,T,*params),Hessian(bounds[1],G,T,*params)]) 
    return H_min

# refine
def local_opt(G,T,params1,params2,x1_0,x2_0,accuracy_x=1e-3,accuracy_common_tangent=1e-3,niter_local=300,niter_local_x=300):
    x1=x1_0
    x2=x2_0
    common_tangent=(G(x1,T,*params1)-G(x2,T,*params2))/(x1-x2)
    dcommon_tangent=1
    niter=1
    while (dcommon_tangent>accuracy_common_tangent) and (niter<=niter_local):    
        def eq1(x):
            y=G(x,T,*params1)-common_tangent*x
            return y
        def eq2(x):
            y=G(x,T,*params2)-common_tangent*x
            return y
        dx=1
        niter_x=1    
        while (dx>accuracy_x) and (niter_x<=niter_local_x):
            dx=-jax.grad(eq1)(x1)/jax.grad(jax.grad(eq1))(x1)
            x1=x1+dx
            niter_x=niter_x+1    
        dx=1
        niter_x=1
        while (dx>accuracy_x) and (niter_x<=niter_local_x):
            dx=-jax.grad(eq2)(x2)/jax.grad(jax.grad(eq2))(x2)
            x2=x2+dx
            niter_x=niter_x+1
        common_tangent_new=(G(x1,T,*params1)-G(x2,T,*params2))/(x1-x2)
        dcommon_tangent=abs(common_tangent_new-common_tangent)
        common_tangent=common_tangent_new
        niter=niter+1
        return x1,x2,common_tangent

def unit_dirac(x):
    if x==0:
        return 1
    else:
        return 0

# params_opt unit (kJ, J/K)
#Data from: Chakrabarti, D. J., and D. E. Laughlin. 
#"The Cu− Rh (Copper-Rhodium) system." Journal of Phase Equilibria 2.4 (1982): 460-462. 
def phase_boundary(params_opt,T,column=[0,1],ngrid=99,niter_global=10,tolerance=1e-5,accuracy_x=1e-3,accuracy_common_tangent=1e-3,niter_local=300,niter_local_x=300):
    a1=params_opt[0]*1000;b1=params_opt[1]*1000;
    g11=0;g21=0
    a2=params_opt[2]*1000;b2=params_opt[3]*1000;
    g12=13054-9.623*T;g22=18753-8.372*T   
    params1=(a1,b1,g11,g21)
    params2=(a2,b2,g12,g22)
    x=jnp.concatenate((jnp.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),jnp.array([1e-7,1-1e-7])))
    sample1=jnp.array([[i,G(i,T,*params1),1] for i in x])
    sample2=jnp.array([[i,G(i,T,*params2),2] for i in x])
    sample=jnp.concatenate((sample1,sample2))
    hull = qhull(sample,column=column,niter_global=niter_global,tolerance=tolerance)
    hul=jnp.array(hull)
    composition_change=abs(hul[:,1,0]-hul[:,0,0])
    hull_phase_boundary = jnp.repeat(hul, composition_change>1/(ngrid+1)+accuracy_x, axis=0)
    if hull_phase_boundary.shape[0]>0:
        phase_boundary_set=[]
        for i in range(hull_phase_boundary.shape[0]):
            hull_phase_boundary1=hull_phase_boundary[i,0,0]
            hull_phase_boundary2=hull_phase_boundary[i,1,0]
            phase1=hull_phase_boundary[i,0,2]
            phase2=hull_phase_boundary[i,1,2]
            p1=unit_dirac(phase1-1)*params1+unit_dirac(phase1-2)*params2
            p2=unit_dirac(phase2-1)*params1+unit_dirac(phase2-2)*params2
            refinement=local_opt(G,T,p1,p2,hull_phase_boundary1,hull_phase_boundary2,accuracy_x=accuracy_x,accuracy_common_tangent=accuracy_common_tangent,niter_local=niter_local,niter_local_x=niter_local_x)
            composition_phase_pairs=[(refinement[0],phase1),(refinement[1],phase2)]
            phase_boundary_set.append(composition_phase_pairs)
        return phase_boundary_set
    else:
        print('No phase boundary!')
        return []

def df(x,G,T,params1,params2,ngrid=99,accuracy_x=1e-7,niter_local_x=300):
    x_metastable_sample=jnp.concatenate((jnp.linspace(1/(ngrid+1),1-1/(ngrid+1),ngrid),jnp.array([1e-7,1-1e-7])))
    if G(x,T,*params1)>G(x,T,*params2):
        params1_old=params1
        params1=params2
        params2=params1_old    
    k=jax.grad(G)(x,T,*params1)
    dist=jnp.array([abs(k*(i-x)-(G(i,T,*params2)-G(x,T,*params1)))/(1+k**2) for i in x_metastable_sample])
    x_metastable=x_metastable_sample[jnp.argmin(dist)]
    def eq(xm):
        tangent=jax.grad(G)(x,T,*params1)
        y=G(xm,T,*params2)-tangent*xm
        return jnp.reshape(y,())
    dx=1
    niter_x=1
    bounds=(1e-7,1.0-1e-7)   
    while (abs(dx)>accuracy_x) and (niter_x<=niter_local_x):
        dx=-jax.grad(eq)(x_metastable)/jax.grad(jax.grad(eq))(x_metastable)
        x_metastable=x_metastable+dx
        if x_metastable<bounds[0]:
            x_metastable=bounds[0]
        if x_metastable>bounds[1]:
            x_metastable=bounds[1]
        niter_x=niter_x+1   
    R=8.314
    driving_force=(G(x_metastable,T,*params2)-G(x,T,*params1)-k*(x_metastable-x))/(R*T)
    return jnp.reshape(driving_force,())

def df_min(params_opt,T,n_df_min=50,*args):
    a1=params_opt[0]*1000;b1=params_opt[1]*1000;
    g11=0;g21=0
    a2=params_opt[2]*1000;b2=params_opt[3]*1000;
    g12=13054-9.623*T;g22=18753-8.372*T 
    params1=(a1,b1,g11,g21)
    params2=(a2,b2,g12,g22)
    x_sample=jnp.concatenate((jnp.linspace(0.1,0.9,9),jnp.array([1e-7,1-1e-7])))
    df_sample=jnp.array([df(i.reshape(1),G,T,params1,params2,*args) for i in x_sample])
    x=x_sample[jnp.argmin(df_sample)]
    df_sample_max=df_sample.max()
    dfp=jax.grad(df)
    dfpp=jax.grad(dfp)    
    dx=1
    niter_x=1
    bounds=(1e-7,1.0-1e-7)  
    while (abs(dx)>1e-3) and (niter_x<=n_df_min):
        dx=-dfp(x,G,T,params1,params2,*args)/dfpp(x,G,T,params1,params2,*args)
        x=x+dx
        if x<bounds[0]:
            x=bounds[0]
        if x>bounds[1]:
            x=bounds[1]
        niter_x=niter_x+1
    driving_force_min=min([df(x,G,T,params1,params2,*args),df(bounds[0],G,T,params1,params2,*args),df(bounds[1],G,T,params1,params2,*args)]) 
    return driving_force_min

#-------------- loss including both phase boundary and driving force 

def ordered_phase_boundary(phase_boundary):
    phase_type=[]
    c=[]
    for j in range(len(phase_boundary)):
        phase_boundary[j]=sorted(phase_boundary[j], key=lambda x: (x[1],x[0]))
        phases=[k[1] for k in phase_boundary[j]]
        compositions=[k[0] for k in phase_boundary[j]]
        if phases in phase_type:
            c[phase_type.index(phases)].append(compositions)
        else:
            phase_type.append(phases)
            c.append([compositions])
    for j in range(len(c)):
        c[j]=sorted(c[j], key=lambda x: x[0])
        c[j]=jnp.array(c[j])
    return phase_type,c

#target_phase_boundary has the form:
#[(T1,x1),...,(Tn,xn)]
from jax.nn import relu
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
    #l_phase_boundary_record.append(alpha*l_phase_boundary)
    #l_driving_force_record.append(l_driving_force)
    #l_immiscibility_record.append(l_immiscibility)
    #l_record.append(l)
    return l

import numpy as np
import time
"""
#params_record=[]
#g_record=[]
grad_1=jax.value_and_grad(loss)
def np_grad_1(params_opt):
    #start_time = time.time()
    print(params_opt)
    f,g=grad_1(params_opt,target_phase_boundary)
    #params_record.append(params_opt)
    #g_record.append(g)
    #step_time = time.time() - start_time
    #print(f'Step time: {step_time} sec')
    return f,np.array(g)
"""

#---save
def data_converter(a):
    try:
        a=a.primal
    except:
        pass
    return a
