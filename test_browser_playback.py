"""Compare browser evolution against the independent NumPy physics reference."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
import browser_playback as playback
import harmonic_ui as ho
import well_ui as well
from physics import gaussian_packet, oscillator_moments


@pytest.mark.parametrize('mode',[m for m in playback.LOCAL_MODES if m not in ('rect','beat','dispersion')])
@pytest.mark.parametrize('time',[0.,.37,3.1])
def test_browser_wavefunction_and_moments_match_numpy(mode,time):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js is required to verify the JavaScript evolution')
    hs=dict(weights=[1.,.4,0.,.8]+[0.]*9,phases=[35.,-20.,0.,90.]+[0.]*9)
    ws=dict(n=3,m=7,weight=.35,phase=47.)
    model=playback.model(mode,2.3,1.7,.45,['real','density','phase'],'fixed',1,hs,ws)
    script="global.window={dash_clientside:{}};eval(require('fs').readFileSync('assets/browser_playback.js','utf8'));const v=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(window.quantumPlayback.frame(v.model,v.t)));"
    process=subprocess.run([node,'-e',script],input=json.dumps(dict(model=model,t=time)),text=True,capture_output=True,check=True,cwd=Path(__file__).parent)
    frame=json.loads(process.stdout);x=np.array(model['x'])
    if mode=='plane':expected=np.exp(1j*(model['k']*x-model['k']**2*time/(2*model['mass'])))
    elif mode=='free':
        sx=1/(2*model['sk']);expected=gaussian_packet(x,time,model['k'],sx,model['mass'])/(2*np.pi*sx*sx)**(-.25)
        assert frame['mean']==pytest.approx(model['k']*time/model['mass'])
        assert frame['sd']==pytest.approx(sx*np.sqrt(1+(time/(2*model['mass']*sx*sx))**2))
    elif mode=='well':
        expected=well.state_data(ws,time)[0];rho=np.abs(expected)**2;mean=np.trapezoid(x*rho,x)
        assert frame['mean']==pytest.approx(mean)
        assert frame['sd']==pytest.approx(np.sqrt(np.trapezoid((x-mean)**2*rho,x)))
        assert frame['norm']==pytest.approx(1.)
    else:
        c=np.array(model['cr'])+1j*np.array(model['ci']);expected=(c*np.exp(-1j*np.array(model['energies'])*time))@ho.BASIS/model['amp']
        moments=oscillator_moments(time,c)
        for a,b in [('mean','mean_x'),('sd','sd_x'),('sdP','sd_p')]:assert frame[a]==pytest.approx(moments[b])
    np.testing.assert_allclose(frame['re'],expected.real,atol=1e-12)
    np.testing.assert_allclose(frame['im'],expected.imag,atol=1e-12)
