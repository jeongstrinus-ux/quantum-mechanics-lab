"""Continuous Fourier evolution and browser rendering of the comparison."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
import dispersion_ui as disp
from physics import gaussian_width

@pytest.mark.parametrize('k,sk,m,t',[(3,.7,1,0),(2,.45,1.5,2.3),(.5,2,4,5)])
def test_continuous_fourier_solutions(k,sk,m,t):
    x=np.linspace(-4,8,17);sx=1/(2*sk);a0=(2*np.pi*sx*sx)**(-.25)
    ks=np.linspace(k-12*sk,k+12*sk,5001)
    a=(2*sx*sx/np.pi)**.25*np.exp(-sx*sx*(ks-k)**2)
    actual=disp.fields(x,t,k,sk,m)
    for wave,omega in zip(actual,[(k/m)*ks,ks**2/(2*m)]):
        integral=np.trapezoid(a[:,None]*np.exp(1j*(ks[:,None]*x-omega[:,None]*t)),ks,axis=0)/np.sqrt(2*np.pi)/a0
        np.testing.assert_allclose(wave,integral,atol=1e-12)
    initial=disp.fields(x-k*t/m,0,k,sk,m)[0]
    np.testing.assert_allclose(actual[0],initial,atol=1e-13)

@pytest.mark.parametrize('t',[0,.37,4.2])
def test_browser_matches_numpy(t):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js needed for browser comparison')
    k,sk,m=2.3,.45,1.7
    model=disp.model(k,sk,m,['real','density','envelope','reference'],'fixed',1)
    traces=[{'meta':{'row':r,'field':f}} for r in (1,2) for f in ('real','density','env','neg-env','reference')]
    base={'data':traces,'layout':{'uirevision':'dispersion','shapes':[{} for _ in range(4)],'annotations':[{} for _ in range(4)]}}
    script="global.window={dash_clientside:{no_update:null}};eval(require('fs').readFileSync('assets/browser_playback.js','utf8'));const v=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(window.quantumPlayback.render(v.t,v.m,v.b,'dispersion')));"
    r=subprocess.run([node,'-e',script],input=json.dumps(dict(t=t,m=model,b=base)),capture_output=True,text=True,check=True,cwd=Path(__file__).parent)
    figure=json.loads(r.stdout)[0];x=disp.X;sx=1/(2*sk)
    for i,wave in enumerate(disp.fields(x,t,k,sk,m)):
        center=k*t/m;width=[sx,gaussian_width(t,sx,m)][i]
        expected=[wave.real,abs(wave)**2,abs(wave),-abs(wave),np.exp(-(x-center)**2/(2*sx*sx))]
        for trace,values in zip(figure['data'][i*5:i*5+5],expected):np.testing.assert_allclose(trace['y'],values,atol=1e-13)
        assert figure['layout']['shapes'][i*2]['x0']==pytest.approx(center)
        assert figure['layout']['shapes'][i*2+1]['x1']==pytest.approx(center+width)
