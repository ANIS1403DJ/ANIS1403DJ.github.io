import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

## parametres de probleme 
lambda_val = 1
U0 = 1
T=60 
## euler explicite avec un pas de temps donné 
Nt = int(input("donner une valeur pour Nt :")) ## nombre de pas 
dt= T/Nt
t= np.linspace(0,T,Nt+1)

## initialisation 
U=np.zeros(Nt+1)
U[0]= U0

## schema d'euler explicite 
for k in range (Nt) :
    U[k+1]= U[k]*(1-lambda_val*dt)

## solution exact 
uext = U0*np.exp(-lambda_val*t)

## erreur en temps
erreur = np.abs(U-uext)
## tracage de solution et erreur 
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t,U,'o-',label='euler explicite')
plt.plot(t,uext,'k-',label='solution exacte ')
plt.xlabel('temps(s)')
plt.ylabel('U(t) ')
plt.title('comparaison des solutions')
plt.legend ()

plt.subplot(1,2,2)
plt.plot(t,erreur ,'r',label=' erreur')
plt.xlabel('temps(s)')
plt.ylabel('erreur')
plt.title('courbe erreur ')
plt.tight_layout()
plt.show()

## partie convergence 
lambda_val = 1
T=60
dts=np.logspace(0,-3,20)   ##20 valeurs de Dt entre 1 et 0.001 
L2_U=[]
L2_dU=[]
for dt in dts:
    t= np.arange(0,T+dt,dt)
    U=np.zeros_like(t)
    U[0]=U0
    for k in range(len(t)-1):
        U[k+1] = U[k] - dt * lambda_val * U[k]
 
 
    U_exact = U0 * np.exp(-lambda_val * t)
    dU_exact = -lambda_val * U_exact
## derrivé numerique 
    dU_num = np.diff(U)/dt
    dU_exact_cut = dU_exact[1:]  # alignement
    
    # Erreur L2
    L2_U.append(np.sqrt(np.mean((U - U_exact)**2)))
    L2_dU.append(np.sqrt(np.mean((dU_num - dU_exact_cut)**2)))
## taracage 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.loglog(dts,L2_U,'o-',label='erreur L2_U')
plt.xlabel('dts')
plt.ylabel('L2_U')
plt.title('erreur L2_U en focntion de dts ')
plt.grid(True)


plt.subplot(1,2,2)
plt.loglog(dts,L2_dU,'k-',label='erreur L2_dU')
plt.xlabel('dts')
plt.ylabel('dL2_U')
plt.title('erreur dL2_U en focntion de dts ')
plt.grid(True)
plt.tight_layout()
plt.show()
