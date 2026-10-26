"""Initial data for exact browser-side evolution; Python remains the reference."""
import numpy as np
import harmonic_ui as ho
import well_ui as well
from physics import sine_basis

LOCAL_MODES=('dispersion','beat','plane','free','well','oscillator','rect')


def model(mode,k,mass,sk,options,axis,zoom,harmonic,well_state,rect_state=None,beat_kind='linear'):
    if mode=='dispersion':
        import dispersion_ui as dispersion
        return dispersion.model(k,sk,mass,options,axis,zoom)
    if mode=='beat':
        import beat_ui as beat
        return dict(mode='beat',kind=beat_kind,k=k,sk=sk,mass=mass,x=beat.X.tolist(),options=options or [],zoom=zoom)
    if mode=='rect':
        import rectangular_ui as rect
        return rect.model(rect_state or rect.DEFAULT_STATE,mass,options,axis,zoom)
    if mode not in LOCAL_MODES:return None
    result=dict(mode=mode,k=k,mass=mass,sk=sk,options=options or [],axis=axis,zoom=zoom)
    if mode=='plane':result['x']=np.linspace(-15,15,1801).tolist()
    elif mode=='free':
        lo=-max(15.,3/sk);hi=max(100.,30*k/mass)
        result['x']=np.linspace(lo,hi,2601).tolist()
    elif mode=='well':
        s=well_state or well.DEFAULT_STATE;n,m=int(s['n']),int(s['m']);q=float(s['weight'])
        c=np.array([1.+0j]) if n==m else np.array([np.sqrt(1-q),np.sqrt(q)*np.exp(1j*np.deg2rad(s['phase']))])
        ns=np.array([n]) if n==m else np.array([n,m])
        result.update(x=well.X.tolist(),basis=sine_basis(well.X,ns,1).tolist(),energies=(ns**2).tolist(),cr=c.real.tolist(),ci=c.imag.tolist(),amp=1.,energy=float(np.abs(c)**2@(ns**2)))
    else:
        s=harmonic or ho.DEFAULT_STATE
        try:c,_,a,_=ho.initial_data(tuple(float(v or 0) for v in s['weights']),tuple(float(v or 0) for v in s['phases']))
        except (ValueError,TypeError,KeyError):return None
        result.update(x=ho.X.tolist(),basis=[row.tolist() if abs(c[n])>0 else [] for n,row in enumerate(ho.basis())],energies=(np.arange(13)+.5).tolist(),cr=c.real.tolist(),ci=c.imag.tolist(),amp=a,energy=float(np.abs(c)**2@(np.arange(13)+.5)))
    return result
