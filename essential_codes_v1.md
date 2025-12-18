# Python Codes for Cosmology Analysis

**Complete collection of Python code for cosmological research and analysis**

---

## Quick Reference

### Installation
```bash
pip install numpy scipy astropy colossus emcee corner
pip install matplotlib pandas scikit-learn
```

### Essential Imports
```python
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy import units as u
import matplotlib.pyplot as plt
```

---

## 1. Basic Cosmological Distances

```python
from astropy.cosmology import FlatLambdaCDM
import numpy as np

# Define cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Redshift array
z = np.array([0.1, 0.5, 1.0, 2.0])

# Calculate distances
d_L = cosmo.luminosity_distance(z)      # Luminosity distance
d_A = cosmo.angular_diameter_distance(z) # Angular diameter distance  
d_C = cosmo.comoving_distance(z)         # Comoving distance

print(f"At z=1:")
print(f"  d_L = {cosmo.luminosity_distance(1.0):.2f}")
print(f"  d_A = {cosmo.angular_diameter_distance(1.0):.2f}")
print(f"  d_C = {cosmo.comoving_distance(1.0):.2f}")
```

---

## 2. Age and Lookback Time

```python
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
z = 1.0

# Age at redshift z
age_z = cosmo.age(z)

# Lookback time
lookback = cosmo.lookback_time(z)

# Age of universe at z=0
age_0 = cosmo.age(0)

print(f"Age at z={z}: {age_z:.2f}")
print(f"Lookback time: {lookback:.2f}")
print(f"Age of universe: {age_0:.2f}")
```

---

## 3. Distance Modulus for Supernovae

```python
from astropy.cosmology import FlatLambdaCDM
import numpy as np

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def distance_modulus(z, M=-19.3):
    """
    Calculate apparent magnitude from absolute magnitude
    m = M + μ(z)
    """
    d_L = cosmo.luminosity_distance(z).value  # Mpc
    mu = 5 * np.log10(d_L) + 25
    return mu, M + mu

# Example: Type Ia supernova
z = np.array([0.5, 1.0, 1.5])
for zi in z:
    mu, m = distance_modulus(zi, M=-19.3)
    print(f"z={zi}: μ={mu:.3f}, m={m:.3f}")
```

---

## 4. Hubble Parameter H(z)

```python
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import matplotlib.pyplot as plt

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

z = np.linspace(0, 5, 100)

# Hubble parameter
H_z = cosmo.H(z)

# Normalized: E(z) = H(z)/H0
E_z = cosmo.efunc(z)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(z, H_z.value, linewidth=2)
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(z, E_z, linewidth=2)
plt.xlabel('Redshift z')
plt.ylabel('E(z) = H(z)/H₀')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5. Power Spectrum with Colossus

```python
from colossus.cosmology import cosmology
import numpy as np
import matplotlib.pyplot as plt

# Set Planck18 cosmology
cosmology.setCosmology('planck18')
cosmo = cosmology.getCurrent()

# Wavenumber array
k = np.logspace(-3, 2, 500)

# Matter power spectrum at z=0
Pk = np.array([cosmo.matterPowerSpectrum(ki, 0) for ki in k])

# Variance
R = np.logspace(-1, 2, 100)
sigma = np.array([cosmo.sigma(r, 0) for r in R])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.loglog(k, Pk, linewidth=2)
ax1.set_xlabel('k [h/Mpc]')
ax1.set_ylabel('P(k) [(Mpc/h)³]')
ax1.grid(True, alpha=0.3, which='both')

ax2.loglog(R, sigma, linewidth=2)
ax2.axvline(8, color='red', linestyle='--', alpha=0.5)
ax2.scatter([8], [cosmo.sigma8], color='red', s=100, zorder=5)
ax2.set_xlabel('R [Mpc/h]')
ax2.set_ylabel('σ(R)')
ax2.set_title(f'σ₈ = {cosmo.sigma8:.3f}')
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()
```

---

## 6. Halo Mass Function

```python
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import numpy as np
import matplotlib.pyplot as plt

cosmology.setCosmology('planck18')
cosmo = cosmology.getCurrent()

# Halo mass range
M = np.logspace(10, 16, 100)

# Mass function at different redshifts
z_vals = [0, 1, 2]
fig = plt.figure(figsize=(10, 6))

for z in z_vals:
    # dn/dM
    dn_dM = mass_function.massFunction(M, z, mdef='200m', model='tinker10')
    plt.loglog(M, dn_dM, label=f'z={z}', linewidth=2)

plt.xlabel('Halo Mass [M☉/h]')
plt.ylabel('dn/dM [Mpc⁻³ h³ (M☉)⁻¹]')
plt.title('Halo Mass Function')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()
```

---

## 7. MCMC Fitting: Supernovae Data

```python
import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import corner

# Mock supernova data
np.random.seed(42)
z_sne = np.linspace(0.01, 2, 150)
cosmo_true = FlatLambdaCDM(H0=70, Om0=0.3)
d_L = cosmo_true.luminosity_distance(z_sne).value
mu_true = 5 * np.log10(d_L) + 25
mu_obs = mu_true + np.random.normal(0, 0.15, len(z_sne))
mu_err = 0.15

def log_likelihood(theta, z, mu, mu_err):
    Om, H0 = theta
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    d_L = cosmo.luminosity_distance(z).value
    mu_model = 5 * np.log10(d_L) + 25
    chi2 = np.sum(((mu - mu_model) / mu_err)**2)
    return -0.5 * chi2

def log_prior(theta):
    Om, H0 = theta
    if 0.1 < Om < 0.5 and 60 < H0 < 80:
        return 0.0
    return -np.inf

def log_probability(theta, z, mu, mu_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, mu_err)

# Run MCMC
nwalkers, ndim, nsteps = 32, 2, 5000
initial = np.array([0.3, 70]) + 0.05 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                 args=(z_sne, mu_obs, mu_err))
sampler.run_mcmc(initial, nsteps, progress=True)

# Extract results
samples = sampler.get_chain(discard=1000, thin=15, flat=True)
fig = corner.corner(samples, labels=['Ωₘ', 'H₀'],
                    truths=[0.3, 70], show_titles=True)
plt.show()

# Print results
for i, label in enumerate(['Ωₘ', 'H₀']):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{label} = {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f}")
```

---

## 8. BAO Analysis

```python
from astropy.cosmology import FlatLambdaCDM
import numpy as np

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Comoving distance
z = np.array([0.35, 0.57])
d_M = cosmo.comoving_distance(z).value  # Mpc

# Hubble distance
c = 299792.458  # km/s
H_z = cosmo.H(z).value
d_H = c / H_z

# Volume-averaged distance
D_V = ((d_M**2 * z * d_H)**(1/3))

# BAO scale
r_d = 149.28  # Mpc/h (from recombination)

print("BAO measurements:")
for i in range(len(z)):
    print(f"  z={z[i]}: D_V = {D_V[i]:.1f} Mpc, D_V/r_d = {D_V[i]/r_d:.2f}")
```

---

## 9. Compare Cosmologies

```python
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck18
import numpy as np
import matplotlib.pyplot as plt

# Different models
cosmo_lcdm = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo_open = LambdaCDM(H0=70, Om0=0.3, Ode0=0.65)
cosmo_planck = Planck18

z = np.linspace(0, 3, 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age
ages = [cosmo_lcdm.age(z), cosmo_open.age(z), cosmo_planck.age(z)]
for i, (cosmo, label) in enumerate([('ΛCDM', cosmo_lcdm), 
                                     ('Open', cosmo_open),
                                     ('Planck18', cosmo_planck)]):
    ages_arr = cosmo.age(z)
    axes[0, 0].plot(z, ages_arr, label=cosmo, linewidth=2)
axes[0, 0].set_ylabel('Age (Gyr)')
axes[0, 0].set_title('Age of Universe')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Luminosity distance
for cosmo, label in [('ΛCDM', cosmo_lcdm), ('Open', cosmo_open)]:
    axes[0, 1].plot(z, cosmo.luminosity_distance(z).value, label=label, linewidth=2)
axes[0, 1].set_ylabel('d_L (Mpc)')
axes[0, 1].set_title('Luminosity Distance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Hubble parameter
for cosmo, label in [('ΛCDM', cosmo_lcdm), ('Open', cosmo_open)]:
    axes[1, 0].plot(z, cosmo.H(z).value, label=label, linewidth=2)
axes[1, 0].set_ylabel('H(z) [km/s/Mpc]')
axes[1, 0].set_title('Hubble Parameter')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Density parameters
for cosmo, label in [('ΛCDM', cosmo_lcdm), ('Open', cosmo_open)]:
    axes[1, 1].plot(z, cosmo.Om(z), label=f'Ωₘ ({label})', linewidth=2)
axes[1, 1].set_ylabel('Ω_m(z)')
axes[1, 1].set_xlabel('Redshift z')
axes[1, 1].set_title('Matter Density Evolution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('Redshift z')

plt.tight_layout()
plt.show()
```

---

## 10. Growth Factor

```python
from scipy.integrate import odeint
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import matplotlib.pyplot as plt

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def growth_ode(y, z):
    D, dD = y
    a = 1 / (1 + z)
    H = cosmo.H(z).value
    dH = (cosmo.H(z + 0.001).value - H) / 0.001
    Om_z = cosmo.Om(z)
    
    d2D = -((2/a + dH/H)) * dD + (3*Om_z/(2*a**3)) * D
    return [dD, d2D]

# Solve ODE
z_vals = np.linspace(1000, 0, 1000)
y0 = [1/(1+1000), -1/(1+1000)**2]
solution = odeint(growth_ode, y0, z_vals)
D = solution[:, 0]

# Normalize
D_norm = D / D[-1]

plt.figure(figsize=(10, 6))
plt.plot(z_vals, D_norm, linewidth=2.5, color='blue')
plt.xlabel('Redshift z')
plt.ylabel('D(z) / D(0)')
plt.title('Linear Growth Factor')
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()
plt.show()

print(f"Growth factor at z=0: {D_norm[-1]:.4f}")
print(f"Growth factor at z=1: {D_norm[500]:.4f}")
```

---

## Additional Resources

**Official Documentation**:
- Astropy: https://docs.astropy.org/en/stable/cosmology/
- Colossus: https://bdiemer.bitbucket.io/colossus/
- emcee: https://emcee.readthedocs.io/

**Key Tutorials**:
- Astropy Cosmology: https://learn.astropy.org/tutorials/redshift-plot.html
- MCMC Basics: https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html

**Research Papers**:
- Planck 2018: https://arxiv.org/abs/1807.06209
- DES Y3 Cosmology: https://arxiv.org/abs/2105.13549
- Pantheon+ SNe: https://arxiv.org/abs/2202.04077

---

**Python 3.10+** | **Astropy 5.0+** | **Colossus 1.3+**
