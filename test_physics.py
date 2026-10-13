import numpy as np
import pytest
from physics import gaussian_packet, gaussian_width, k_density, sampled_superposition, progressive_superposition, mirror_fields, mirror_component, moments


@pytest.mark.parametrize('t,m,sigma,k0',[(0,1,.8,3),(4,1,.8,3),(12,.5,.25,6),(8,4,2,1)])
def test_gaussian_normalization_motion_and_width(t,m,sigma,k0):
    center=k0*t/m;sd=gaussian_width(t,sigma,m)
    x=np.linspace(center-12*sd,center+12*sd,20001)
    norm,mean,width=moments(x,np.abs(gaussian_packet(x,t,k0,sigma,m))**2)
    assert norm==pytest.approx(1,abs=1e-10)
    assert mean==pytest.approx(center,abs=1e-9)
    assert width==pytest.approx(sd,rel=1e-9)
    assert width/(2*sigma)>=.5


def test_gaussian_solves_schrodinger_equation():
    x=np.linspace(-5,8,1000);t=2.;eps=1e-4;k0=2.;sx=.9;m=1.4
    psi=gaussian_packet(x,t,k0,sx,m)
    dt=(gaussian_packet(x,t+eps,k0,sx,m)-gaussian_packet(x,t-eps,k0,sx,m))/(2*eps)
    dxx=(gaussian_packet(x+eps,t,k0,sx,m)-2*psi+gaussian_packet(x-eps,t,k0,sx,m))/eps**2
    assert np.max(np.abs(1j*dt+dxx/(2*m)))<2e-7


def test_fourier_quadrature_agrees_with_analytic_gaussian():
    x=np.linspace(-8,8,801);sk=.7
    psi,ks,amps=sampled_superposition(x,101,3,sk)
    exact=gaussian_packet(x,0,3,1/(2*sk))
    assert np.max(np.abs(psi-exact))<.004
    k=np.linspace(3-10*sk,3+10*sk,10001)
    norm,mean,sd=moments(k,k_density(k,3,sk))
    assert (norm,mean,sd)==pytest.approx((1,3,sk),abs=1e-8)


@pytest.mark.parametrize('k,sigma',[(.5,4),(3,2.5),(6,1)])
def test_mirror_boundary_no_transmission_energy_conservation(k,sigma):
    x=np.linspace(-100,0,20001)
    integrator=np.trapezoid
    energies=[]
    for t in [0,9,18,27,36,40]:
        fields=mirror_fields(x,t,k,sigma)
        assert mirror_fields(np.array([0]),t,k,sigma)['electric'][0]==0
        assert np.all(mirror_fields(np.array([.1,2]),t,k,sigma)['energy']==0)
        assert np.allclose(fields['energy'],(fields['electric']**2+fields['magnetic']**2)/2)
        energies.append(integrator(fields['energy'],x))
    assert max(energies)-min(energies)<1e-9


def test_reflection_reverses_direction_and_e_phase():
    x=np.linspace(-30,-6,1001)
    start=mirror_fields(x,0,3,2.5)
    end=mirror_fields(x,36,3,2.5)
    # At the symmetric return position, the real Gaussian pulse has inverted E.
    assert np.allclose(end['reflected'],-start['incident'],atol=1e-12)
    assert np.mean(start['flux'])>0
    assert np.mean(end['flux'])<0


def test_progressive_n_adds_modes_and_changes_packet():
    x=np.linspace(-15,15,1801)
    few,kfew,_=progressive_superposition(x,7,3,.7)
    many,kmany,_=progressive_superposition(x,61,3,.7)
    assert np.allclose(kfew,kmany[27:34])
    assert kmany[-1]-kmany[0]>kfew[-1]-kfew[0]
    assert np.max(np.abs(few-many))>.3
    full,_,_=progressive_superposition(x,101,3,.7)
    assert np.max(np.abs(full-gaussian_packet(x,0,3,1/1.4)))<.004


@pytest.mark.parametrize('k,t',[(.3,0),(1,8),(3,18),(5.4,29)])
def test_each_mirror_fourier_component_satisfies_boundary(k,t):
    p=mirror_component(np.array([0.,2.]),t,k,2.5,3)
    assert p['incident'][0]==pytest.approx(-p['reflected'][0],abs=1e-14)
    assert p['electric'][0]==pytest.approx(0,abs=1e-14)
    assert p['electric'][1]==0


def test_mirror_component_integral_reconstructs_real_pulse():
    # Positive-frequency cosine weights include both Gaussian carrier lobes.
    sigma=2.;k0=1.;t=7.;x=np.linspace(-24,0,41)
    ks=np.linspace(0,5,4001)
    w=sigma/np.sqrt(np.pi)*(np.exp(-sigma*sigma*(ks-k0)**2)+np.exp(-sigma*sigma*(ks+k0)**2))
    incident=np.trapezoid(w*np.cos(np.outer(x-t+18,ks)),ks,axis=1)
    reflected=-np.trapezoid(w*np.cos(np.outer(-x-t+18,ks)),ks,axis=1)
    exact=mirror_fields(x,t,k0,sigma)
    assert np.allclose(incident,exact['incident'],atol=1e-10)
    assert np.allclose(reflected,exact['reflected'],atol=1e-10)


@pytest.mark.parametrize('t,k,sx,m',[(0,3,.714,1),(6,3,.714,1),(12,3,.714,1),(36,.5,.25,.5),(288,.5,3.333,4)])
def test_quantum_wall_normalization_and_boundary(t,k,sx,m):
    from physics import wall_packet
    width=gaussian_width(t,sx,m)
    left=abs(-18+k*t/m)+18+12*width
    x=np.linspace(-left,0,40001)
    f=wall_packet(x,t,k,sx,m)
    assert np.trapezoid(f['density'],x)==pytest.approx(1,abs=1e-8)
    assert f['psi'][-1]==0
    assert np.all(wall_packet(np.array([.01,2]),t,k,sx,m)['density']==0)


def test_quantum_wall_schrodinger_and_momentum_reversal():
    from physics import wall_packet
    from app import wall_frame
    x=np.linspace(-30,-.1,1201);eps=1e-4;t=5.;k=3.;sx=1/1.4
    psi=wall_packet(x,t,k,sx)['psi']
    dt=(wall_packet(x,t+eps,k,sx)['psi']-wall_packet(x,t-eps,k,sx)['psi'])/(2*eps)
    dxx=(wall_packet(x+eps,t,k,sx)['psi']-2*psi+wall_packet(x-eps,t,k,sx)['psi'])/eps**2
    assert np.max(np.abs(1j*dt+dxx/2))<5e-7
    before=wall_frame(3,.7,1,0);after=wall_frame(3,.7,1,12)
    assert before['mean_k']>2.9 and after['mean_k']<-2.9
    assert after['sd']>before['sd']*8
    assert np.trapezoid(before['prob'],before['k'])==pytest.approx(1,abs=1e-8)
    assert after['spectral_norm']==pytest.approx(1,abs=1e-10)
    # The plotted k interval omits a small nonzero Fourier tail from the wall.
    assert .9999 < np.trapezoid(after['prob'],after['k']) <= 1


@pytest.mark.parametrize('t',[0,6,12])
def test_wall_plane_mode_integral_matches_packet(t):
    from physics import wall_plane_mode,wall_packet
    x=np.linspace(-30,0,41);ks=np.linspace(0,10,3001)
    values=np.array([wall_plane_mode(x,t,float(k),3,.7) for k in ks])
    reconstructed=np.trapezoid(values,ks,axis=0)
    exact=wall_packet(x,t,3,1/1.4)['psi']
    assert np.max(np.abs(reconstructed-exact))<1e-9
    assert all(abs(wall_plane_mode([0],t,float(k),3,.7)[0])<1e-14 for k in [1.6,2.3,3,3.7,4.4])


@pytest.mark.parametrize('k,sk,m',[(.5,2,.5),(6,2,4),(.5,.15,4)])
def test_wall_input_extremes_and_phase_updates(k,sk,m):
    from app import wall_frame,figures,wall_frame_patches,configure
    frame=wall_frame(k,sk,m,0)
    assert frame['norm']==pytest.approx(1,abs=1e-8)
    assert frame['mean_k']==pytest.approx(k,abs=1e-5)
    options=['density','phase']
    fig=figures('wall',k,sk,7,m,2,0,options)[0]
    assert len(fig.data)==2
    assert np.array_equal(fig.data[-1].marker.color,np.angle(frame['fields']['psi']))
    assert len(figures('wall',k,sk,7,m,2,0,options)[1].data)==1
    assert not any(o['value'].startswith('k') for o in configure('wall')[0])
    patch=wall_frame_patches(k,sk,m,1,options,'auto',1)[0].to_plotly_json()
    assert any(op['location']==['data',1,'marker','color'] for op in patch['operations'])


@pytest.mark.parametrize('k,sk,m',[(3,.7,1),(.5,2,.5),(6,.15,4)])
def test_wall_end_reaches_outgoing_mean_minus_40(k,sk,m):
    from app import wall_end_time,wall_frame
    end=wall_end_time(k,sk,m)
    frame=wall_frame(k,sk,m,end)
    assert frame['mean']==pytest.approx(-40,abs=2e-4)
    assert frame['mean_k']<0
    assert frame['norm']==pytest.approx(1,abs=1e-8)


def test_oscillator_basis_orthonormal_and_stationary():
    from physics import oscillator_basis,oscillator_packet
    x=np.linspace(-10,10,10001);basis=oscillator_basis(x,12)
    gram=basis@basis.T*(x[1]-x[0])
    assert np.max(np.abs(gram-np.eye(13)))<1e-10
    for n in [0,1,6,12]:
        c=np.zeros(13,complex);c[n]=1
        initial=oscillator_packet(x,0,c);later=oscillator_packet(x,1.7,c)
        assert np.allclose(np.abs(initial)**2,np.abs(later)**2,atol=1e-12)


def test_oscillator_packet_schrodinger_and_exact_moments():
    from physics import oscillator_coefficients,oscillator_packet,oscillator_moments
    c=oscillator_coefficients([1,2,3,4],[0,45,-75,120])
    x=np.linspace(-9,9,12001);t=.8;eps=1e-4
    psi=oscillator_packet(x,t,c)
    dt=(oscillator_packet(x,t+eps,c)-oscillator_packet(x,t-eps,c))/(2*eps)
    dxx=(oscillator_packet(x+eps,t,c)-2*psi+oscillator_packet(x-eps,t,c))/eps**2
    assert np.max(np.abs(1j*dt+dxx/2-x*x*psi/2))<1e-6
    norm,mean,sd=moments(x,np.abs(psi)**2);exact=oscillator_moments(t,c)
    assert (norm,mean,sd)==pytest.approx((1,exact['mean_x'],exact['sd_x']),abs=1e-9)


def test_oscillator_parity_phase_and_recurrence():
    from physics import oscillator_coefficients,oscillator_packet,oscillator_moments
    c=oscillator_coefficients([1,1],[0,0]);t=.7
    assert oscillator_moments(t,c)['mean_x']==pytest.approx(np.cos(t)/np.sqrt(2))
    changed=oscillator_coefficients([1,1],[0,90])
    assert oscillator_moments(0,changed)['mean_x']==pytest.approx(0,abs=1e-12)
    assert oscillator_moments(0,changed)['mean_p']==pytest.approx(1/np.sqrt(2))
    breathing=oscillator_coefficients([1,0,1],[0,0,0])
    assert oscillator_moments(t,breathing)['mean_x']==pytest.approx(0)
    assert abs(oscillator_moments(0,breathing)['sd_x']-oscillator_moments(np.pi/2,breathing)['sd_x'])>.4
    x=np.linspace(-8,8,2001)
    assert np.allclose(oscillator_packet(x,2*np.pi,c),-oscillator_packet(x,0,c),atol=1e-12)
    assert np.allclose(oscillator_packet(x,4*np.pi,c),oscillator_packet(x,0,c),atol=1e-12)


def test_oscillator_coherent_preset_and_invalid_weights():
    from harmonic_ui import preset_values
    from physics import oscillator_coefficients,oscillator_moments
    values=preset_values('coherent');c=oscillator_coefficients(values[:13],values[13:])
    for t in [0,.5,2,4]:
        result=oscillator_moments(t,c)
        assert result['norm']==pytest.approx(1,abs=1e-12)
        assert result['mean_x']==pytest.approx(np.sqrt(2)*1.5*np.cos(t),abs=2e-5)
        assert result['sd_x']==pytest.approx(1/np.sqrt(2),abs=5e-5)
        assert result['sd_p']==pytest.approx(1/np.sqrt(2),abs=5e-5)
    with pytest.raises(ValueError):oscillator_coefficients([0,0],[0,0])
    with pytest.raises(ValueError):oscillator_coefficients([-1,1],[0,0])


def test_oscillator_time_limit_is_two_cycles():
    from app import simulation_end
    assert simulation_end('oscillator',3,1,.7)==pytest.approx(4*np.pi)


@pytest.mark.parametrize('options',[['density','phase','potential'],['real','imag','density','reference','phase','potential']])
def test_oscillator_time_patch_matches_full_figure(options):
    from app import figures,metric
    from harmonic_ui import DEFAULT_STATE,time_patch
    t=.7
    figure=figures('oscillator',3,.7,7,1,2,t,options,'auto')[0]
    patch,_=time_patch(DEFAULT_STATE,t,options,'auto',1,metric)
    for op in patch.to_plotly_json()['operations']:
        loc=op['location']
        assert loc[:2]!=['layout','yaxis2']
        if len(loc)==3 and loc[0]=='data' and loc[2]=='y':
            assert np.allclose(op['params']['value'],figure.data[loc[1]].y,equal_nan=True)
    assert figure.data[-1].yaxis in (None,'y')
    assert figure.layout.shapes[0].y0==pytest.approx(1)
    assert figure.layout.yaxis.range[1]<5


def test_oscillator_new_state_renders_at_zero_before_slider_reset(monkeypatch):
    from types import SimpleNamespace
    import app
    state=dict(weights=[1,1]+[0]*11,phases=[0,90]+[0]*11)
    monkeypatch.setattr(app,'ctx',SimpleNamespace(triggered_id='ho-state',triggered_prop_ids={'ho-state.data':'ho-state'}))
    outputs=app.render('oscillator',3,.7,7,1,2,.8,['density','phase'],state,'fixed',1)
    assert outputs[-1]=='0.00'
    assert outputs[2][0].children[1].children=='0.000'


@pytest.mark.parametrize('n',[0,4,12])
def test_oscillator_waveform_baseline_is_mean_energy(n):
    from app import figures
    state=dict(weights=[int(i==n) for i in range(13)],phases=[0]*13)
    for t in [0,.7]:
        fig=figures('oscillator',3,.7,7,1,2,t,['density','phase','potential'],harmonic_state=state)[0]
        assert fig.layout.shapes[0].y0==pytest.approx(n+.5)
        assert fig.layout.shapes[0].y1==pytest.approx(n+.5)
        density=fig.data[0]
        assert np.allclose(np.asarray(density.y[:2401])-(n+.5),density.customdata[:2401])
        assert np.allclose(density.y[2401:],n+.5)


@pytest.mark.parametrize('length',[4.,10.,20.])
def test_sine_basis_orthogonality_and_projection(length):
    from physics import sine_basis,sine_projection
    x=np.linspace(0,length,4097);b=sine_basis(x,np.arange(1,41),length)
    assert np.max(np.abs(b@b.T*(x[1]-x[0])-np.eye(40)))<1e-12
    f=.4*b[0]-.7*b[3]+.2j*b[6]
    coefficients,_=sine_projection(x,f,40,length)
    expected=np.zeros(40,complex);expected[[0,3,6]]=[.4,-.7,.2j]
    assert np.allclose(coefficients,expected,atol=1e-12)


@pytest.mark.parametrize('shape',['gaussian','triangle','offcenter','rectangle','double'])
def test_sine_reconstruction_error_and_retained_norm(shape):
    from fourier_ui import discrete_data
    x,f,c,b=discrete_data(10.,shape,.1)
    assert np.trapezoid(f*f,x)==pytest.approx(1,abs=1e-12)
    previous=1.
    for n in [1,3,7,30,60]:
        partial=c[:n]@b[:n]
        error=np.trapezoid(np.abs(f-partial)**2,x)
        assert error==pytest.approx(1-np.sum(np.abs(c[:n])**2),abs=1e-12)
        assert error<=previous+1e-12
        previous=error
    assert abs(partial[0])<1e-12 and abs(partial[-1])<1e-12


@pytest.mark.parametrize('sx,k0,x0',[(.3,3,1.2),(1.5,-2,-3)])
def test_continuous_gaussian_forward_and_inverse_fourier(sx,k0,x0):
    from physics import gaussian_fourier_amplitude
    x=np.linspace(-30,30,20001);ks=np.linspace(-8,8,41)
    psi=gaussian_packet(x,0,k0,sx,x0=x0)
    numerical=np.trapezoid(np.exp(-1j*np.outer(ks,x))*psi,x,axis=1)/np.sqrt(2*np.pi)
    assert np.allclose(numerical,gaussian_fourier_amplitude(ks,k0,sx,x0),atol=1e-11)
    k=np.linspace(-30,30,20001);xx=np.linspace(-6,6,41)
    a=gaussian_fourier_amplitude(k,k0,sx,x0)
    inverse=np.trapezoid(np.exp(1j*np.outer(xx,k))*a,k,axis=1)/np.sqrt(2*np.pi)
    assert np.allclose(inverse,gaussian_packet(xx,0,k0,sx,x0=x0),atol=1e-11)
    assert np.trapezoid(np.abs(a)**2,k)==pytest.approx(1,abs=1e-12)
    assert np.allclose(np.abs(a)**2,np.abs(gaussian_fourier_amplitude(k,k0,sx,0))**2)


@pytest.mark.parametrize('n,m,q,phase',[(1,2,.5,0),(3,8,.3,75),(8,1,1,180),(4,4,.5,180)])
def test_infinite_well_boundaries_norm_revival_and_energy(n,m,q,phase):
    from well_ui import X,state_data
    state=dict(n=n,m=m,weight=q,phase=phase)
    initial,_,indices,prob=state_data(state,0)
    for t in [0,.17,2.3,2*np.pi]:
        psi,_,_,_=state_data(state,t)
        assert abs(psi[0])<1e-12 and abs(psi[-1])<1e-12
        assert np.trapezoid(abs(psi)**2,X)==pytest.approx(1,abs=1e-12)
        # H/E1 = -(1/pi²)d²/dξ²; verify by kinetic energy quadrature.
        cgrad=np.gradient(psi,X,edge_order=2)
        energy=np.trapezoid(abs(cgrad)**2,X)/np.pi**2
        assert energy==pytest.approx(prob@indices**2,rel=1e-4)
    assert np.max(abs(state_data(state,2*np.pi)[0]-initial))<1e-12


def test_infinite_well_stationary_density_and_interference():
    from well_ui import state_data
    state=dict(n=1,m=2,weight=0,phase=0)
    assert np.max(abs(abs(state_data(state,.4)[0])**2-abs(state_data(state,0)[0])**2))<1e-12
    state['weight']=.5
    assert np.max(abs(abs(state_data(state,np.pi/3)[0])**2-abs(state_data(state,0)[0])**2))>1


def test_build_single_mode_keeps_gaussian_weight_and_comparison():
    from app import figures,configure
    x=np.linspace(-15,15,1801);k0=3.;sk=.7
    single,ks,amps=progressive_superposition(x,1,k0,sk)
    _,ks3,amps3=progressive_superposition(x,3,k0,sk)
    assert ks[0]==ks3[1]
    assert amps[0]==amps3[1]
    expected=np.sqrt(k_density(np.array([k0]),k0,sk))[0]*8*sk/100/np.sqrt(2*np.pi)
    assert np.allclose(single,expected*np.exp(1j*k0*x))
    result=figures('build',k0,sk,1,1,2,0,configure('build')[1])
    assert result[8]=='평면파를 더해 파동묶음 만들기'
    assert any('모든 k' in trace.name for trace in result[0].data)
    assert any('N=1개' in trace.name for trace in result[0].data)
    assert np.allclose(result[-1]['density'],abs(single)**2)


@pytest.mark.parametrize('shape',['gaussian','triangle','offcenter','rectangle','double'])
def test_ranked_fourier_reconstruction_preserves_sign_and_minimizes_error(shape):
    from fourier_ui import discrete_data,ranked_reconstruction
    x,f,c,basis=discrete_data(10.,shape,.1)
    previous=float('inf')
    for count in [1,3,7,30,60]:
        partial,selected,order=ranked_reconstruction(c,basis,count)
        assert np.all(np.diff(np.abs(c[order]))<=1e-14)
        assert np.allclose(partial,c[selected]@basis[selected])
        error=np.trapezoid((f-partial)**2,x)
        assert error==pytest.approx(1-np.sum(np.abs(c[selected])**2),abs=1e-12)
        assert error<=previous+1e-12
        assert error<=np.trapezoid((f-c[:count]@basis[:count])**2,x)+1e-12
        previous=error
    assert np.allclose(partial,c@basis)
