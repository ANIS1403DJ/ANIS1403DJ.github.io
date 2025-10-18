import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

## paramètres du problème 
lambda_val = 1
U0 = 1
T = 60 

## euler explicite avec un pas de temps donné 
Nt = int(input("donner une valeur pour Nt :"))  # nombre de pas 
dt = T / Nt
t = np.linspace(0, T, Nt+1)

## initialisation 
U = np.zeros(Nt+1)
U[0] = U0

## schéma d'Euler explicite 
for k in range(Nt):
    U[k+1] = U[k] * (1 - lambda_val * dt)

## solution exacte 
uext = U0 * np.exp(-lambda_val * t)

## erreur en temps
erreur = np.abs(U - uext)

## Tracé de la solution numérique et exacte
plt.figure(figsize=(8,5))
plt.plot(t, U, 'o-', label='Euler explicite')
plt.plot(t, uext, 'k-', label='Solution exacte')
plt.xlabel('temps (s)')
plt.ylabel('U(t)')
plt.title('Solution numérique vs solution exacte')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("solution_euler.png", dpi=300)  # sauvegarde solution
plt.show()

## Tracé de l'erreur
plt.figure(figsize=(8,5))
plt.plot(t, erreur, 'r', label='Erreur')
plt.xlabel('temps (s)')
plt.ylabel('Erreur')
plt.title('Erreur en fonction du temps')
plt.grid(True)
plt.tight_layout()
plt.savefig("erreur_euler.png", dpi=300)  # sauvegarde erreur
plt.show()

## partie convergence 
lambda_val = 1
T = 60
dts = np.logspace(0, -3, 20)   # 20 valeurs de dt entre 1 et 0.001 
L2_U = []
L2_dU = []

for dt in dts:
    t = np.arange(0, T+dt, dt)
    U = np.zeros_like(t)
    U[0] = U0
    for k in range(len(t)-1):
        U[k+1] = U[k] - dt * lambda_val * U[k]

    U_exact = U0 * np.exp(-lambda_val * t)
    dU_exact = -lambda_val * U_exact

    # dérivée numérique 
    dU_num = np.diff(U) / dt
    dU_exact_cut = dU_exact[1:]  # alignement

    # Erreur L2
    L2_U.append(np.sqrt(np.mean((U - U_exact)**2)))
    L2_dU.append(np.sqrt(np.mean((dU_num - dU_exact_cut)**2)))

## Tracé erreur L2 de la solution
plt.figure(figsize=(8,5))
plt.loglog(dts, L2_U, 'o-', label='Erreur L2_U')
plt.xlabel('dt')
plt.ylabel('L2_U')
plt.title('Erreur L2_U en fonction de dt')
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("convergence_L2_U.png", dpi=300)  # sauvegarde L2_U
plt.show()

## Tracé erreur L2 de la dérivée
plt.figure(figsize=(8,5))
plt.loglog(dts, L2_dU, 'k-', label='Erreur L2_dU')
plt.xlabel('dt')
plt.ylabel('L2_dU')
plt.title('Erreur L2_dU en fonction de dt')
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("convergence_L2_dU.png", dpi=300)  # sauvegarde L2_dU
plt.show()


 
   
   





   



