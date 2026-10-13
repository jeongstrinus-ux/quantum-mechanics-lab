"""Separate matter-wave and classical electromagnetic models.

Matter waves: hbar=1; sigma_x and sigma_k refer to probability SDs.
Mirror: c=1, E0=1, impedance-scaled h=Z H; x<=0 is air, x=0 is PEC.
Matter-wave wall: exact half-line image solution with Dirichlet boundary.
No finite dielectric step, absorption or finite-box dynamics.
"""
from __future__ import annotations
import numpy as np


def gaussian_packet(x, t: float, k0: float, sigma_x: float, mass: float = 1.0, x0: float = 0.0):
    """Exact normalized free-particle Gaussian, not finite-window normalized."""
    x = np.asarray(x, dtype=float)
    b = t / (2 * mass * sigma_x**2)
    z = 1 + 1j * b
    center = x0 + k0 * t / mass
    psi = (2 * np.pi * sigma_x**2)**(-0.25) / np.sqrt(z)
    psi = psi * np.exp(-(x - center)**2 / (4 * sigma_x**2 * z))
    return psi * np.exp(1j * (k0 * (x - x0) - k0**2 * t / (2 * mass)))


def gaussian_width(t: float, sigma_x: float, mass: float = 1.0):
    return sigma_x * np.sqrt(1 + (t / (2 * mass * sigma_x**2))**2)


def wall_packet(x,t,k0,sigma_x,mass=1.,x0=-18.):
    """Exactly normalized Dirichlet image solution on the left half-line.

    The antisymmetric image evolves with the same free Hamiltonian. This is an
    infinite potential wall, not a dielectric or a light-wave boundary.
    """
    x=np.asarray(x,dtype=float)
    normalization=np.sqrt(-np.expm1(-x0*x0/(2*sigma_x*sigma_x)-2*sigma_x*sigma_x*k0*k0))
    incoming=gaussian_packet(x,t,k0,sigma_x,mass,x0)/normalization
    reflected=-gaussian_packet(-x,t,k0,sigma_x,mass,x0)/normalization
    incoming=np.where(x<=0,incoming,0j)
    reflected=np.where(x<=0,reflected,0j)
    psi=incoming+reflected
    return dict(psi=psi,incident=incoming,reflected=reflected,density=np.abs(psi)**2)


def wall_plane_mode(x,t,k,k0,sigma_k,mass=1.,dk=1.,x0=-18.):
    """One positive-k wall eigenmode, formed from a +k/-k plane-wave pair.

    Integrating these terms over all k>0 reconstructs wall_packet. dk supplies
    the quadrature bin width. The mode itself is global and is not a packet.
    """
    x=np.asarray(x,dtype=float)
    sx=1/(2*sigma_k)
    norm=np.sqrt(-np.expm1(-x0*x0/(2*sx*sx)-2*sx*sx*k0*k0))
    ak=np.sqrt(k_density(k,k0,sigma_k))*np.exp(-1j*k*x0)
    aminus=np.sqrt(k_density(-k,k0,sigma_k))*np.exp(1j*k*x0)
    coefficient=(ak-aminus)*np.exp(-1j*k*k*t/(2*mass))*dk/(np.sqrt(2*np.pi)*norm)
    return np.where(x<=0,coefficient*(np.exp(1j*k*x)-np.exp(-1j*k*x)),0j)


def k_density(k, k0: float, sigma_k: float):
    k = np.asarray(k, dtype=float)
    return np.exp(-0.5 * ((k - k0) / sigma_k)**2) / (np.sqrt(2 * np.pi) * sigma_k)


def sampled_superposition(x, count: int, k0: float, sigma_k: float):
    """Trapezoidal Fourier quadrature over ±4 sigma_k at t=0.

    Few modes produce periodic replicas: do not treat window moments as full-line
    uncertainty. A single plane wave has illustrative amplitude one.
    """
    x = np.asarray(x, dtype=float)
    if count == 1:
        return np.exp(1j * k0 * x), np.array([k0]), np.array([1.0])
    ks = np.linspace(k0 - 4 * sigma_k, k0 + 4 * sigma_k, count)
    amps = np.sqrt(k_density(ks, k0, sigma_k))
    weights = np.ones(count)
    weights[[0, -1]] = 0.5
    weights *= (ks[1] - ks[0]) / np.sqrt(2 * np.pi)
    return np.exp(1j * np.outer(x, ks)) @ (amps * weights), ks, amps


def progressive_superposition(x, count: int, k0: float, sigma_k: float):
    """Add nested central modes at a fixed spacing for the classroom N control.

    101 modes cover k0±4 sigma_k. Increasing N adds outer modes instead of
    just refining quadrature. These are finite, periodic illustrative sums;
    their spatial moments are not full-line quantum uncertainties.
    """
    count=int(count)
    if count<1 or count>101 or count%2!=1:
        raise ValueError('count must be an odd integer between 1 and 101')
    x=np.asarray(x,dtype=float)
    dk=8*sigma_k/100
    ks=k0+np.arange(-(count//2),count//2+1)*dk
    amps=np.sqrt(k_density(ks,k0,sigma_k))
    return np.exp(1j*np.outer(x,ks)) @ (amps*dk/np.sqrt(2*np.pi)),ks,amps


def mirror_component(x,t,k,sigma,k0,x0=-18.0):
    """Positive-k Fourier cosine component of the real Gaussian carrier pulse.

    f(z)=integral_0^infinity B(k) cos(k(z-x0)) dk. Return waves at k with
    amplitude B(k)/B(k0), NOT a narrow-band pulse with an added envelope.
    Individual plane waves are global; only their full sum is localized.
    """
    x=np.asarray(x,dtype=float)
    def weight(q):
        return sigma/np.sqrt(np.pi)*(np.exp(-sigma*sigma*(q-k0)**2)+np.exp(-sigma*sigma*(q+k0)**2))
    amplitude=float(weight(k)/weight(k0))
    ei=np.where(x<=0,amplitude*np.cos(k*(x-t-x0)),0)
    er=np.where(x<=0,-amplitude*np.cos(k*(-x-t-x0)),0)
    return dict(incident=ei,reflected=er,electric=ei+er,amplitude=amplitude)


def mirror_fields(x, t: float, k0: float, sigma: float, x0: float = -18.0):
    """Exact 1D nondispersive PEC reflection of a real Gaussian EM pulse.

    E_i=f(x-t), E_r=-f(-x-t). H_r has the opposite propagation sign.
    E(0,t)=0. u=(E²+(ZH)²)/2=Ei²+Er²; net flux S=E*(ZH).
    The initial Gaussian has a negligible image tail in air, retained exactly.
    """
    x = np.asarray(x, dtype=float)
    def f(z):
        return np.exp(-((z - x0) / sigma)**2 / 4) * np.cos(k0 * (z - x0))
    inside = x <= 0
    ei = np.where(inside, f(x - t), 0.0)
    er = np.where(inside, -f(-x - t), 0.0)
    e = ei + er
    h = ei - er
    return dict(incident=ei, reflected=er, electric=e, magnetic=h,
                energy=ei**2 + er**2, flux=e * h)


def mirror_envelopes(x, t, sigma, x0=-18.0):
    x = np.asarray(x, dtype=float)
    inside = x <= 0
    return (np.where(inside, np.exp(-((x - t - x0) / sigma)**2 / 4), 0),
            np.where(inside, np.exp(-((-x - t - x0) / sigma)**2 / 4), 0))


def moments(x, density):
    """Numerical moments used for checks, not clipped classroom uncertainty."""
    integral = getattr(np, "trapezoid", np.trapz if hasattr(np, "trapz") else None)
    norm = float(integral(density, x))
    mean = float(integral(density * x, x) / norm)
    sd = float(np.sqrt(integral(density * (x - mean)**2, x) / norm))
    return norm, mean, sd


def oscillator_basis(x,max_n=12):
    """Normalized Hermite functions for hbar=m=omega=1 via stable recurrence."""
    x=np.asarray(x,dtype=float)
    basis=[np.pi**(-.25)*np.exp(-x*x/2)]
    if max_n>=1:basis.append(np.sqrt(2)*x*basis[0])
    for n in range(1,max_n):
        basis.append(np.sqrt(2/(n+1))*x*basis[n]-np.sqrt(n/(n+1))*basis[n-1])
    return np.asarray(basis)


def oscillator_coefficients(weights,phase_degrees):
    weights=np.asarray(weights,dtype=float);angles=np.asarray(phase_degrees,dtype=float)
    if weights.ndim!=1 or len(weights)!=len(angles) or not len(weights):
        raise ValueError('가중치와 위상 입력을 확인하세요.')
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(angles)) or np.any(weights<0):
        raise ValueError('가중치는 0 이상의 유한한 값이어야 합니다.')
    if not np.any(weights>0):raise ValueError('적어도 한 상태의 가중치를 0보다 크게 입력하세요.')
    scaled=weights/weights.max()
    return np.sqrt(scaled/scaled.sum())*np.exp(1j*np.deg2rad(angles))


def oscillator_packet(x,t,coefficients):
    c=np.asarray(coefficients,dtype=complex)
    energies=np.arange(len(c))+.5
    return (c*np.exp(-1j*energies*t))@oscillator_basis(x,len(c)-1)


def oscillator_moments(t,coefficients):
    """Exact moments from ladder operators, independent of the plot window."""
    c=np.asarray(coefficients,dtype=complex)*np.exp(-1j*(np.arange(len(coefficients))+.5)*t)
    n=np.arange(len(c));number=float(np.sum(n*np.abs(c)**2))
    amean=np.sum(np.sqrt(n[1:])*np.conj(c[:-1])*c[1:])
    a2=np.sum(np.sqrt(n[2:]*(n[2:]-1))*np.conj(c[:-2])*c[2:])
    mean_x=float(np.sqrt(2)*amean.real);mean_p=float(np.sqrt(2)*amean.imag)
    var_x=max(0.,number+.5+a2.real-mean_x**2)
    var_p=max(0.,number+.5-a2.real-mean_p**2)
    return dict(mean_x=mean_x,mean_p=mean_p,sd_x=np.sqrt(var_x),sd_p=np.sqrt(var_p),
                energy=number+.5,norm=float(np.sum(np.abs(c)**2)))


def sine_basis(x,indices,length):
    """Real orthonormal Dirichlet basis on [0,L], n=1,2,..."""
    x=np.asarray(x,dtype=float);n=np.asarray(indices,dtype=int)
    return np.sqrt(2/length)*np.sin(np.outer(n,np.pi*x/length))


def sine_projection(x,target,count,length):
    basis=sine_basis(x,np.arange(1,count+1),length)
    return np.trapezoid(basis*np.asarray(target),x,axis=1),basis


def gaussian_fourier_amplitude(k,k0,sigma_x,x0=0.):
    """Unitary Fourier transform of gaussian_packet(x,0,...,x0)."""
    k=np.asarray(k,dtype=float)
    return (2*sigma_x**2/np.pi)**.25*np.exp(-sigma_x**2*(k-k0)**2)*np.exp(-1j*k*x0)
