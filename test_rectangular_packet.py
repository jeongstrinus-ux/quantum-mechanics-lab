"""Independent Fourier/propagator checks for the textbook rectangle."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from physics import rectangular_spectrum,rectangular_packet,fresnel_cs
import rectangular_ui as rect


@pytest.mark.parametrize('a',[.5,2.,6.])
@pytest.mark.parametrize('k0',[0.,2.,-2.])
def test_rectangle_transform_matches_direct_integral(a,k0):
    nodes,w=np.polynomial.legendre.leggauss(192);x=a*nodes
    k=k0+np.array([-7.,-np.pi,0.,np.pi,5.])/a
    reference=a/np.sqrt(4*np.pi*a)*(np.exp(-1j*np.outer(k-k0,x))@w)
    np.testing.assert_allclose(rectangular_spectrum(k,a,k0),reference,atol=2e-13)
    assert abs(rectangular_spectrum(k0+np.pi/a,a,k0))<1e-14
    q=np.linspace(-2000,2000,160001)
    assert np.trapezoid(np.abs(rectangular_spectrum(k0+q/a,a,k0))**2,q/a)==pytest.approx(1,abs=2e-4)


@pytest.mark.parametrize('time',[.15,1.,8.])
def test_rectangle_matches_independent_propagator(time):
    a=1.2;k0=1.7;mass=1.3;x=np.linspace(-4,4,31)
    nodes,w=np.polynomial.legendre.leggauss(256);initial=a*nodes
    kernel=np.sqrt(mass/(2j*np.pi*time))*np.exp(1j*mass*(x[:,None]-initial)**2/(2*time))
    reference=a/np.sqrt(2*a)*(kernel@(w*np.exp(1j*k0*initial)))
    np.testing.assert_allclose(rectangular_packet(x,time,a,k0,mass),reference,atol=3e-12)


def test_rectangle_symmetry_translation_and_initial_state():
    a=2.;k0=2.;mass=1.4;t=3.;y=np.linspace(-30,30,1201)
    resting=rectangular_packet(y,t,a,0,mass)
    moving=rectangular_packet(y+k0*t/mass,t,a,k0,mass)
    np.testing.assert_allclose(np.abs(moving)**2,np.abs(resting)**2,atol=2e-14)
    np.testing.assert_allclose(np.abs(resting)**2,np.abs(resting[::-1])**2,atol=1e-14)
    np.testing.assert_allclose(rectangular_packet([-3,-1,0,1,3],0,a),[0,.5,.5,.5,0])


def test_rectangle_satisfies_free_schrodinger_equation():
    x=np.linspace(-3,3,51);t=.8;dx=1e-4;dt=1e-5;mass=1.3
    f=lambda xx,tt:rectangular_packet(xx,tt,1.1,.7,mass)
    time_derivative=(f(x,t+dt)-f(x,t-dt))/(2*dt)
    laplacian=(f(x+dx,t)-2*f(x,t)+f(x-dx,t))/dx**2
    np.testing.assert_allclose(1j*time_derivative,-laplacian/(2*mass),atol=2e-6)


@pytest.mark.parametrize('time',[0.,.01,.4,5.])
def test_browser_rectangle_matches_full_line_numpy(time):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js is required for the browser cross-check')
    model=rect.model(dict(a=.75,k0=2.),1.7,['density'],'fixed',1)
    script="global.window={dash_clientside:{}};eval(require('fs').readFileSync('assets/browser_playback.js','utf8'));let v=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(window.quantumPlayback.frame(v.m,v.t)));"
    result=subprocess.run([node,'-e',script],input=json.dumps(dict(m=model,t=time)),text=True,capture_output=True,check=True,cwd=Path(__file__).parent)
    frame=json.loads(result.stdout)
    reference=rectangular_packet(model['x'],time,model['a'],model['k'],model['mass'])*np.sqrt(2*model['a'])
    np.testing.assert_allclose(frame['re'],reference.real,atol=5e-9)
    np.testing.assert_allclose(frame['im'],reference.imag,atol=5e-9)
    assert frame['mean']==pytest.approx(model['k']*time/model['mass'])
    assert frame['sd'] is None  # Do not report a finite-window variance as Δx.


def test_rectangle_controls_and_time_limit_are_connected():
    import app
    controls=app.configure('rect')
    assert len(controls[0])==5
    assert controls[7]=={}  # Mass is visible.
    assert controls[10]=={'display':'none'}  # Unrelated Gaussian k0 is hidden.
    first=app.configure_time_limit('rect',3,1,.7,{},None,dict(a=1.,k0=2.))
    second=app.configure_time_limit('rect',3,1,.7,{},None,dict(a=2.,k0=2.))
    assert first['end']==second['end']
    assert first!=second  # Width changes must reset the browser clock too.
    assert first['end']*2==24  # Moving center reaches the viewport edge.


def test_real_rectangle_and_boost_are_distinct_initial_conditions():
    import app
    real=app.collect_rectangle_state(2.,2.,'real')
    boost=app.collect_rectangle_state(2.,2.,'boost')
    assert real[0]['k0']==0. and real[3]=={'display':'none'}
    assert boost[0]['k0']==2. and boost[3]=={}
    x=np.linspace(-1.9,1.9,101)
    p0=rectangular_packet(x,0,2.,real[0]['k0'])
    p1=rectangular_packet(x,0,2.,boost[0]['k0'])
    np.testing.assert_allclose(p0,.5)
    np.testing.assert_allclose(np.abs(p0)**2,np.abs(p1)**2,atol=1e-15)
    assert np.max(np.abs(p1.imag))>.4
    eq0=rect.equations(real[0]);eq1=rect.equations(boost[0])
    assert 'e^{ik_0x}' not in eq0[1].children
    assert 'e^{ik_0x}' in eq1[1].children
