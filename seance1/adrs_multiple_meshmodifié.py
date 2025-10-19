import math
import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL PARAMETERS
K = 0.1
L = 1.0
Time = 20.
V = 1
lamda = 1

# NUMERICAL PARAMETERS
NX = 10
NT = 10000
ifre = 1000000
eps = 0.001
niter_refinement = 10

errorL2 = np.zeros(niter_refinement)
errorH1 = np.zeros(niter_refinement)
semiH2 = np.zeros(niter_refinement)
itertab = np.zeros(niter_refinement)

for iter in range(niter_refinement):
    NX += 5
    dx = L / (NX - 1)
    dt = dx**2 / (V * dx + 4 * K + dx**2)
    itertab[iter] = dx

    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    RHS = np.zeros(NX)
    Tex = np.zeros(NX)

    for j in range(1, NX - 1):
        Tex[j] = np.exp(-20 * (j * dx - 0.5)**2)
    for j in range(1, NX - 1):
        Tx = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)
        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / dx**2
        F[j] = V * Tx - K * Txx + lamda * Tex[j]

    n = 0
    res = 1
    res0 = 1
    while n < NT and res / res0 > eps:
        n += 1
        res = 0
        for j in range(1, NX - 1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
            Txx = (T[j - 1] - 2 * T[j] + T[j + 1]) / dx**2
            RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + F[j])
            res += abs(RHS[j])
        for j in range(1, NX - 1):
            T[j] += RHS[j]
            RHS[j] = 0
        if n == 1:
            res0 = res

    errH1h = 0
    errL2h = 0
    semih2 = 0
    for j in range(1, NX - 1):
        Texx = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)
        Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
        errL2h += dx * (T[j] - Tex[j])**2
        errH1h += dx * (Tx - Texx)**2
        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / dx**2
        semih2 += dx * Txx**2

    errorL2[iter] = errL2h
    errorH1[iter] = errL2h + errH1h
    semiH2[iter] = semih2

# === IDENTIFICATION DES PARAMÈTRES C ET k (sans sklearn) ===
log_h = np.log(itertab)
log_err = np.log(np.sqrt(errorL2) / semiH2)

A = np.vstack([log_h, np.ones_like(log_h)]).T
k, log_C = np.linalg.lstsq(A, log_err, rcond=None)[0]
C = np.exp(log_C)
print(f"Identifié : C = {C:.4e}, k = {k:.4f}")

# === COURBES DE CONVERGENCE ===
plt.figure()
plt.loglog(itertab, np.sqrt(errorL2) / semiH2, 'o-', label=r"$\|u - u_h\|_{L^2} / \|u\|_{H^2}$")
plt.loglog(itertab, C * itertab**k, '--', label=fr"$C h^{{{k:.2f}}}$")
plt.loglog(itertab, C * itertab**(k + 1), ':', label=fr"$C h^{{{k+1:.2f}}}$")
plt.xlabel("h")
plt.ylabel("Erreur relative")
plt.title("Convergence en espace")
plt.legend()
plt.grid(True)
plt.show()