import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================
# PHYSICAL PARAMETERS
# =============================
K = 0.1
L = 1.0
Time = 20.0
V = 1.0
lamda = 1.0

# =============================
# NUMERICAL PARAMETERS
# =============================
NX = 10                # nombre de points de maillage initial
NT = 10000             # nombre max d'itérations temporelles
ifre = 1000000         # (non utilisé)
eps = 0.001            # critère d'arrêt (résidu relatif)
niter_refinement = 10  # nombre de raffinements de maillage

# Tableaux pour stocker les erreurs
errorL2 = np.zeros(niter_refinement)
errorH1 = np.zeros(niter_refinement)
semiH2 = np.zeros(niter_refinement)
itertab = np.zeros(niter_refinement)

interpL2 = np.zeros(niter_refinement)
interpH1 = np.zeros(niter_refinement)

# =============================
# BOUCLE SUR LES RAFFINEMENTS
# =============================
for iter in range(niter_refinement):
    NX += 5
    dx = L / (NX - 1)
    dt = dx**2 / (V * dx + 4 * K + dx**2)
    itertab[iter] = dx

    # Grille et variables
    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)
    Tex = np.zeros(NX)
    F = np.zeros(NX)
    RHS = np.zeros(NX)
    rest = []

    # -----------------------------
    # Solution exacte et source F
    # -----------------------------
    for j in range(1, NX - 1):
        Tex[j] = np.exp(-20 * (x[j] - 0.5) ** 2)

    for j in range(1, NX - 1):
        Tx = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)
        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / dx**2
        F[j] = V * Tx - K * Txx + lamda * Tex[j]

    # -----------------------------
    # Boucle temporelle
    # -----------------------------
    n = 0
    res = 1.0
    res0 = 1.0

    while n < NT and res / res0 > eps:
        n += 1
        res = 0.0

        for j in range(1, NX - 1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
            Txx = (T[j - 1] - 2 * T[j] + T[j + 1]) / dx**2
            RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + F[j])
            res += abs(RHS[j])

        for j in range(1, NX - 1):
            T[j] += RHS[j]

        if n == 1:
            res0 = res
        rest.append(res)

    # -----------------------------
    # Calcul des erreurs numériques
    # -----------------------------
    errL2h = 0.0
    errH1h = 0.0
    semih2 = 0.0

    for j in range(1, NX - 1):
        Texx = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)   # approx dérivée
        Tx = (T[j + 1] - T[j - 1]) / (2 * dx)

        errL2h += dx * (T[j] - Tex[j])**2
        errH1h += dx * (Tx - Texx)**2

        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / dx**2
        semih2 += dx * Txx**2

    errorL2[iter] = errL2h
    errorH1[iter] = errL2h + errH1h
    semiH2[iter] = semih2

    # -----------------------------
    # Calcul des erreurs d’interpolation
    # -----------------------------
    errL2_interp = 0.0
    errH1_interp = 0.0

    for j in range(1, NX - 2):
        x_mid = x[j] + dx / 2
        u_exact_mid = np.exp(-20 * (x_mid - 0.5) ** 2)
        u_interp_mid = (Tex[j] + Tex[j + 1]) / 2

        dTex_dx = -40 * (x_mid - 0.5) * u_exact_mid
        dInterp_dx = (Tex[j + 1] - Tex[j]) / dx

        errL2_interp += dx * (u_exact_mid - u_interp_mid)**2
        errH1_interp += dx * (dTex_dx - dInterp_dx)**2

    interpL2[iter] = errL2_interp
    interpH1[iter] = errL2_interp + errH1_interp

# =============================
# RÉGRESSION LOG-LOG POUR ESTIMER C ET k
# =============================
def fit_model(h, C, k):
    return C * h**k

h = itertab
sqrtL2 = np.sqrt(errorL2)
sqrtH1 = np.sqrt(errorH1)
sqrtInterpL2 = np.sqrt(interpL2)
sqrtInterpH1 = np.sqrt(interpH1)

popt_L2, _ = curve_fit(fit_model, h, sqrtL2)
popt_H1, _ = curve_fit(fit_model, h, sqrtH1)
popt_L2i, _ = curve_fit(fit_model, h, sqrtInterpL2)
popt_H1i, _ = curve_fit(fit_model, h, sqrtInterpH1)

print(f"[Erreur L2 num] : k = {popt_L2[1]:.3f}, C = {popt_L2[0]:.3e}")
print(f"[Erreur H1 num] : k = {popt_H1[1]:.3f}, C = {popt_H1[0]:.3e}")
print(f"[Erreur L2 interp] : k = {popt_L2i[1]:.3f}, C = {popt_L2i[0]:.3e}")
print(f"[Erreur H1 interp] : k = {popt_H1i[1]:.3f}, C = {popt_H1i[0]:.3e}")

# =============================
# TRACÉS DES ERREURS
# =============================
plt.figure()
plt.loglog(h, sqrtL2, 'o-', label='Erreur L2 (num)')
plt.loglog(h, sqrtH1, 's-', label='Erreur H1 (num)')
plt.loglog(h, sqrtInterpL2, 'o--', label='Erreur L2 (interp)')
plt.loglog(h, sqrtInterpH1, 's--', label='Erreur H1 (interp)')

plt.xlabel('dx')
plt.ylabel('Erreur')
plt.title('Erreurs en fonction du pas de maillage')
plt.legend()
plt.grid(True)
plt.show()
