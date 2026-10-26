"""Beat identity, tracked speeds and the browser playback agree."""
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import beat_ui as beat

@pytest.mark.parametrize('k,dk,m,t',[(3,.15,1,0),(2,.7,1.5,1.2),(.5,2,4,3)])
@pytest.mark.parametrize('kind',['linear','matter'])
def test_beat_identity_and_browser(k,dk,m,t,kind):
    x=beat.X;p=beat.parameters(k,dk,m,kind)
    y1,y2,total,env=beat.frame(x,t,k,dk,m,kind)
    product=2*np.cos((dk*x-p['dw']*t)/2)*np.cos(p['kbar']*x-p['wbar']*t)
    np.testing.assert_allclose(total,product,atol=5e-14)
    assert np.all(np.abs(total)<=env+1e-13)
    assert p['vg']==pytest.approx(1 if kind=='linear' else p['kbar']/m)
    assert dk*p['vg']-p['dw']==pytest.approx(0,abs=1e-14)
    node=shutil.which('node')
    if not node:pytest.skip('Node.js required for browser comparison')
    model=dict(mode='beat',kind=kind,k=k,sk=dk,mass=m,x=x.tolist(),options=['envelope'])
    base=dict(layout=dict(uirevision='beat',shapes=[{},{},{},{}],annotations=[{} for _ in range(6)]),data=[{} for _ in range(5)]+[{'name':'마루 추적 1'},{'name':'마루 추적 2'}])
    script="global.window={dash_clientside:{no_update:null}};eval(require('fs').readFileSync('assets/browser_playback.js','utf8'));const a=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(window.quantumPlayback.render(a.t,a.m,a.b,'beat')));"
    r=subprocess.run([node,'-e',script],input=json.dumps(dict(t=t,m=model,b=base)),capture_output=True,text=True,check=True,cwd=Path(__file__).parent)
    fig=json.loads(r.stdout)[0]
    for trace,expected in zip(fig['data'],[y1,y2,total,env,-env]):np.testing.assert_allclose(trace['y'],expected,atol=1e-13)
    assert fig['layout']['shapes'][0]['x0']==pytest.approx(p['vg']*t)
    assert fig['layout']['shapes'][1]['x0']==pytest.approx(p['vp']*t)

    for i in (1,2):
        crest=p[f'w{i}']/p[f'k{i}']*t
        assert fig['data'][4+i]['x']==pytest.approx([crest])
        assert fig['data'][4+i]['y']==[1]
        assert fig['layout']['shapes'][i+1]['x0']==pytest.approx(crest)
        assert np.cos(p[f'k{i}']*crest-p[f'w{i}']*t)==pytest.approx(1.)


def test_nondispersive_waveform_translates_without_deformation():
    x=beat.X;t=1.7
    initial=beat.frame(x-t,0,3,.7,1,'linear')
    later=beat.frame(x,t,3,.7,1,'linear')
    for a,b in zip(initial,later):np.testing.assert_allclose(a,b,atol=1e-13)
