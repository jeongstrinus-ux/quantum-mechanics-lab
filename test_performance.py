"""Regressions for inactive experiments allocating or recalculating resources."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import app
import browser_playback as playback
import fourier_ui as ft
import harmonic_ui as ho
from physics import sine_projection


def test_oscillator_basis_is_not_computed_on_app_import():
    result=subprocess.run([sys.executable,'-c',"import app,harmonic_ui as h;print(h.basis.cache_info().currsize)"],text=True,capture_output=True,check=True,cwd=Path(__file__).parent)
    assert result.stdout.strip()=='0'


def test_inactive_figures_are_not_calculated(monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('inactive experiment was evaluated')
    for module in (ft,ho,app.well,app.rect,app.beat,app.dispersion):monkeypatch.setattr(module,'figures',unexpected)
    monkeypatch.setattr(ft,'vector_intro',unexpected)
    result=app.figures('plane',3,.7,7,1,2.5,0,['real','density'])
    assert len(result[0].data)==2


def test_vector_webgl_is_mounted_only_when_visible(monkeypatch):
    calls=[]
    monkeypatch.setattr(ft,'vector_intro',lambda:calls.append('mount') or 'figure')
    assert app.configure_vector_intro('well','orthogonal',[]) is app.no_update
    assert app.configure_vector_intro('fourier','continuous',[]) is app.no_update
    assert not calls
    assert app.configure_vector_intro('fourier','orthogonal',[])=='figure'
    assert app.configure_vector_intro('fourier','orthogonal',['figure']) is app.no_update
    assert calls==['mount']
    assert app.configure_vector_intro('plane','orthogonal',['figure'])==[]


def test_fourier_targets_share_basis_without_changing_coefficients():
    ft.discrete_data.cache_clear();ft.discrete_basis.cache_clear()
    x,f,c,basis=ft.discrete_data(10.,'gaussian',.1)
    _,_,_,other=ft.discrete_data(10.,'double',.12)
    assert basis is other
    expected,_=sine_projection(x,f,60,10.)
    np.testing.assert_array_equal(c,expected)


def test_oscillator_payload_omits_zero_weight_basis_rows():
    model=playback.model('oscillator',3,1,.7,['density'],'fixed',1,ho.DEFAULT_STATE,None)
    assert sum(bool(row) for row in model['basis'])==2
    full=dict(model,basis=ho.basis().tolist())
    assert len(json.dumps(model))<.25*len(json.dumps(full))
