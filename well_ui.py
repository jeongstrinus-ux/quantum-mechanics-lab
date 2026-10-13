"""Infinite square well: normalized stationary states and two-state evolution."""
import numpy as np
import plotly.graph_objects as go
from dash import html,dcc
from physics import sine_basis
from harmonic_ui import PHASE_COLORS
from teaching_content import question_items,math_block

X=np.linspace(0,1,1601)
DEFAULT_STATE=dict(n=1,m=2,weight=.5,phase=0.)
FIELDS=['n','m','weight','phase']


def controls():
    children=[html.H3('우물 안의 초기 상태'),html.P('양쪽 벽 x=0, L에서 Ψ=0. n=1이 바닥상태입니다.',className='hint')]
    for key,title,lo,hi,step in [('n','첫 상태 n',1,8,1),('m','둘째 상태 m',1,8,1),('weight','둘째 상태의 확률 가중치 q',0,1,.05),('phase','초기 상대 위상 θ (°)',-180,180,15)]:
        slider=dcc.Slider(id='well-'+key,min=lo,max=hi,step=step,value=DEFAULT_STATE[key],marks=None)
        if key in ('n','m'):
            slider=html.Div([
                html.Button('−',id='well-'+key+'-minus',type='button',**{'aria-label':title+' 1 감소'}),
                slider,
                html.Button('+',id='well-'+key+'-plus',type='button',**{'aria-label':title+' 1 증가'})
            ],className='well-step-control')
        children.append(html.Div([html.Div([html.Label(title),html.Output(id='well-'+key+'-value',className='value')],className='labelrow'),slider],className='control'))
    children += [html.P('q=0 또는 1: 에너지 고유상태 하나. 0<q<1: 두 상태의 중첩. 서로 다른 n과 m을 선택하세요.',className='hint'),dcc.Store(id='well-state',data=DEFAULT_STATE)]
    return html.Div(children,id='well-controls',style={'display':'none'})


def state_data(state,t):
    n,m=int(state['n']),int(state['m']);q=float(state['weight']);theta=np.deg2rad(float(state['phase']))
    if n==m:
        # Same eigenstate is one direction, not two orthogonal states.
        c=np.array([1.+0j]);indices=np.array([n])
    else:
        c=np.array([np.sqrt(1-q),np.sqrt(q)*np.exp(1j*theta)]);indices=np.array([n,m])
    basis=sine_basis(X,indices,1.)
    psi=(c*np.exp(-1j*indices**2*t))@basis
    ref=c@basis
    return psi,ref,indices,np.abs(c)**2


def figures(state,t,options,axis_mode,zoom,styled_fig,line,metric,fit_axis):
    psi,ref,indices,prob=state_data(state,t);opts=set(options or [])
    pos=styled_fig('well','위치 x / L','무차원 파동함수 √L Ψ<br>무차원 확률밀도 L |Ψ|²',540)
    spec=styled_fig('well-energy','에너지 Eₙ / E₁','성분 확률 |cₙ|²',360)
    if 'real' in opts:line(pos,X,psi.real,'Re[√L Ψ]','#48d7ff')
    if 'imag' in opts:line(pos,X,psi.imag,'Im[√L Ψ]','#76a9ff',dash='dot')
    if 'density' in opts:line(pos,X,np.abs(psi)**2,'무차원 확률밀도 L |Ψ|²','#ff7daf',fill=True)
    if 'reference' in opts:line(pos,X,np.abs(ref)**2,'t=0 확률밀도','#74e0aa',dash='dash')
    if 'phase' in opts:
        amp=np.abs(psi)
        pos.add_trace(go.Scatter(x=X,y=np.where(amp>.01,amp,np.nan),mode='markers',name='무차원 진폭 √L |Ψ| · 색 = 위상',marker=dict(size=4,color=np.angle(psi),colorscale=PHASE_COLORS,cmin=-np.pi,cmax=np.pi,colorbar=dict(title='위상',tickvals=[-np.pi,0,np.pi],ticktext=['−π','0','π'])),customdata=np.angle(psi),hovertemplate='x/L=%{x:.3f}<br>√L |Ψ|=%{y:.3f}<br>위상=%{customdata:.3f} rad<extra></extra>'))
    rho=np.abs(psi)**2;mean=float(np.trapezoid(X*rho,X));sd=np.sqrt(np.trapezoid((X-mean)**2*rho,X));energy=float(prob@indices**2)
    pos.add_vline(x=0,line_color='#ffd166',line_width=3);pos.add_vline(x=1,line_color='#ffd166',line_width=3)
    pos.add_vline(x=mean,line_color='white',line_dash='dash');pos.add_annotation(x=mean,y=.98,yref='paper',text='⟨x⟩',showarrow=False,xanchor='left',xshift=8)
    pos.update_xaxes(range=[-.03,1.03]);pos.update_yaxes(range=[-2.2/zoom,4.4/zoom] if opts & {'real','imag'} else [-.15/zoom,4.4/zoom])
    if axis_mode=='auto':
        ys=np.concatenate([np.asarray(tr.y) for tr in pos.data]) if pos.data else np.array([0,1])
        ys=ys[np.isfinite(ys)];lo=min(0,float(ys.min()));hi=max(.5,float(ys.max()));pos.update_yaxes(range=[(lo-.15)/zoom,(hi+.2)/zoom])
    levels=np.arange(1,9)
    spec.add_trace(go.Bar(x=levels**2,y=[float(prob[indices==v].sum()) for v in levels],width=1.2,name='에너지 측정 확률',marker_color='#ffd166'))
    spec.update_xaxes(range=[0,66],tickvals=levels**2);spec.update_yaxes(range=[0,1.05]);spec.add_vline(x=energy,line_dash='dash',line_color='white');spec.add_annotation(x=energy,y=1.02,yref='paper',text='⟨E⟩',showarrow=False)
    concepts=[dcc.Markdown('에너지 고유상태는 $\\hat H \\psi_n=E_n\\psi_n$을 만족하는 상태입니다. 시간에 무관한 퍼텐셜에서 $\\Psi_n(x,t)=\\psi_n(x)e^{-iE_nt/\\hbar}$로 진화하며 확률밀도가 일정하므로 정상상태(stationary state)라고 합니다. 정상은 입자가 고전적으로 멈춰 있다는 뜻이 아닙니다.',mathjax=True),math_block('\\psi_n(x)=\\sqrt{\\frac{2}{L}}\\sin\\left(\\frac{n\\pi x}{L}\\right)','E_n=\\frac{n^2\\pi^2\\hbar^2}{2mL^2}=n^2E_1,\\quad n=1,2,\\ldots'),math_block('\\begin{aligned}\\Psi(x,t)={}&\\sqrt{1-q}\\,\\psi_n(x)e^{-iE_nt/\\hbar}\\\\ &+\\sqrt q\\,e^{i\\theta}\\psi_m(x)e^{-iE_mt/\\hbar}\\end{aligned}\\quad(n\\ne m)'),dcc.Markdown('0<x<L에서 V=0, 양쪽 바깥에서 V=∞입니다. 두 경계조건이 허용 파수 $k_n=\\frac{n\\pi}{L}$와 이산 에너지를 정합니다. 벽 하나에서의 반사와 다른 계입니다.',mathjax=True),dcc.Markdown('길이 단위 L, 에너지 단위 $E_1=\\frac{\\pi^2\\hbar^2}{2mL^2}$, 시간 단위 ℏ/E₁을 사용합니다. 화면의 시간은 E₁t/ℏ입니다. 그래프는 √L Ψ와 L|Ψ|²이며 확률밀도의 면적은 1입니다.',mathjax=True),dcc.Markdown('n≠m이면 직교성 때문에 성분 확률은 1−q, q이고 $\\frac{\\langle E\\rangle}{E_1}=(1-q)n^2+qm^2$입니다. 같은 양자수를 고르면 해당 에너지 고유상태 하나로 표시합니다.',mathjax=True),html.P('각 에너지 고유상태 하나의 시간 진화는 정상상태입니다. 서로 다른 에너지의 상태를 중첩하면 합성 상태는 일반적으로 정상상태가 아니며 상대 위상 변화에 따라 위치 확률밀도가 변합니다. 색은 합성 파동함수의 위상이며 벽과 영점에서는 위상이 정의되지 않습니다.'),math_block('\\begin{aligned}|\\Psi|^2={}&(1-q)|\\psi_n|^2+q|\\psi_m|^2\\\\ &+2\\sqrt{q(1-q)}\\,\\psi_n\\psi_m\\\\ &\\quad\\times\\cos\\left[\\theta-\\frac{(E_m-E_n)t}{\\hbar}\\right]\\end{aligned}\\quad(n\\ne m)'),dcc.Markdown('마지막 항이 간섭항입니다. 직교성 $\\int_0^L\\psi_n^*(x)\\psi_m(x)\\,dx=0$은 전체 구간에서의 상쇄를 뜻하며 각 위치에서 $\\psi_n^*(x)\\psi_m(x)=0$이라는 뜻이 아닙니다. 따라서 총확률은 1로 보존되면서 위치 확률밀도는 변할 수 있습니다. 확률밀도만 따로 더하는 통계적 혼합에는 이 간섭항이 없습니다.',mathjax=True),html.P([html.A('모형 근거: MIT · Infinite square well and time evolution',href='https://live.ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/e275e8c6ec64b860d5c2926c0b507c4e_MIT8_04S13_Lec06.pdf',target='_blank')],className='mini')]
    warning='n=m: 같은 에너지 고유상태 하나를 표시합니다. 중첩에 따른 간섭을 보려면 서로 다른 양자수를 선택하세요.' if state['n']==state['m'] else ''
    metrics=[metric('평균 위치 ⟨x⟩ / L',f'{mean:.3f}'),metric('위치 표준편차 Δx / L',f'{sd:.3f}'),metric('평균 에너지 / E₁',f'{energy:.3f}'),metric('확률 총합',f'{np.trapezoid(rho,X):.5f}')]
    return pos,spec,metrics,concepts,question_items('well',0),warning,'x/L 축에서 확률밀도의 면적이 확률이 되도록 L|Ψ|²을 표시합니다. 파동함수는 √LΨ로 표시하며 두 값 모두 단위가 없습니다. 양쪽 노란 선은 무한 벽, 흰 점선은 평균 위치입니다. x축과 초기 기준 y축은 재생 중 고정됩니다.','Eₙ/E₁=n². 막대 높이는 이산 에너지 측정 확률이며 시간에 따라 변하지 않습니다.','무한 퍼텐셜 우물 · 에너지 고유상태와 중첩',dict(x_over_L=X,real_sqrt_L=psi.real,imag_sqrt_L=psi.imag,density_L=rho)
