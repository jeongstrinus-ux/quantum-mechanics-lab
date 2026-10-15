"""Textbook free-particle rectangle: sinc spectrum and Galilean translation."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc,html
from teaching_content import math_block
from harmonic_ui import PHASE_COLORS
from physics import rectangular_packet,rectangular_spectrum,fresnel_browser_table

DEFAULT_STATE=dict(a=2.,k0=0.,kind='real')


def controls():
    rows=[html.H3('교과서 예시 · 사각 파동묶음'),
          html.Label('초기 파동함수 선택'),
          dcc.RadioItems(id='rect-kind',options=[{'label':'1. 실수 사각형 · k 중심 0','value':'real'},{'label':'2. 사각 진폭 + 공간 위상','value':'boost'}],value='real',className='ft-shapes'),
          html.P('두 경우는 초기 확률밀도 |Ψ|²가 같아도 파동함수 Ψ의 공간 위상이 다릅니다.',className='hint')]
    for key,label,lo,hi,step,hint in [('a','초기 사각형의 반폭 a · 단위 L₀',.5,6,.25,'전체 폭은 2a입니다. a를 줄이면 초기 위치 분포가 좁아집니다.'),('k0','파수 분포의 대칭 중심 k₀ · 단위 1/L₀',-4,4,.25,'0: 좌우로 대칭인 퍼짐 · 양수: 대칭 중심이 오른쪽으로 이동 · 음수: 왼쪽으로 이동')]:
        rows.append(html.Div([html.Div([html.Label(label),html.Output(id='rect-'+key+'-value',className='value')],className='labelrow'),dcc.Slider(id='rect-'+key,min=lo,max=hi,step=step,value=2. if key=='k0' else DEFAULT_STATE[key],marks=None),html.P(hint,className='hint')],className='control',id='rect-'+key+'-control',style={'display':'none'} if key=='k0' else {}))
    rows.extend([html.P('추천 순서: 실수 사각형에서 a를 비교한 뒤, 공간 위상을 곱한 초기조건을 선택하고 k₀=+2, −2를 비교하세요.',className='hint'),dcc.Store(id='rect-state',data=DEFAULT_STATE)])
    return html.Div(rows,id='rect-controls',style={'display':'none'})


def viewport(a):
    span=24.
    # Include the exact edges and points just outside them in the t=0 drawing.
    return np.unique(np.r_[np.linspace(-span,span,1201),-a-1e-8,-a,a,a+1e-8])


def end_time(state,mass):
    a=float(state['a']);k=float(state['k0']);span=24.
    return span*mass/abs(k) if abs(k)>1e-12 else 3*mass*a*a


def model(state,mass,options,axis,zoom):
    a=float(state['a']);k=float(state['k0'])
    return dict(mode='rect',a=a,k=k,mass=mass,x=viewport(a).tolist(),options=options or [],axis=axis,zoom=zoom,fresnel=fresnel_browser_table())


def equations(state=None):
    state=state or DEFAULT_STATE
    if state.get('kind','boost' if state.get('k0',0) else 'real')=='real':
        return [html.Strong('초기조건 1 · 파동함수가 실수 상수인 사각형'),
                math_block(r'\Psi(x,0)=\begin{cases}\dfrac{1}{\sqrt{2a}},&|x|<a\\0,&|x|\ge a\end{cases}'),
                math_block(r'\phi(k)=\frac{1}{\sqrt{2\pi}}\int_{-a}^{a}\frac{e^{-ikx}}{\sqrt{2a}}\,dx=\sqrt{\frac{a}{\pi}}\frac{\sin(ak)}{ak}'),
                dcc.Markdown(r'이 초기조건은 파동함수 자체가 실수인 사각형입니다. $\phi(k)$와 $|\phi(k)|^2$는 모두 $k=0$을 중심으로 좌우 대칭이며, 위치 분포의 대칭 중심도 0에 머뭅니다. $\phi(k)$는 진폭, $|\phi(k)|^2$는 확률밀도입니다.',mathjax=True,className='equation-note')]
    return [html.Strong('초기조건 2 · 공간 위상을 곱한 파동함수'),math_block(r'\Psi(x,0)=\begin{cases}\dfrac{e^{ik_0x}}{\sqrt{2a}},&|x|<a\\0,&|x|\ge a\end{cases}'),
            math_block(r'\phi(k)=\frac{1}{\sqrt{2\pi}}\int_{-a}^{a}\frac{e^{ik_0x}}{\sqrt{2a}}e^{-ikx}\,dx=\sqrt{\frac{a}{\pi}}\,\frac{\sin[a(k-k_0)]}{a(k-k_0)}'),
            dcc.Markdown(r'$\phi(k)$는 **푸리에 진폭**이며 파수 확률밀도는 $|\phi(k)|^2$입니다. $k=k_0$에서는 $\sin z/z\to1$의 극한을 사용합니다. 실수 사각형과 **확률밀도는 같지만 파동함수는 다른 초기조건**입니다. 공간 위상 $e^{ik_0x}$에 의해 파수 분포의 중심이 $k_0$로 이동합니다. $k_0=0$으로 설정하면 실수 사각형과 같아집니다.',mathjax=True,className='equation-note')]


def figures(state,mass,t,options,axis,zoom,styled_fig,line,metric,fit_axis):
    a=float(state['a']);k0=float(state['k0']);x=viewport(a);opts=set(options or [])
    a0=1/np.sqrt(2*a);psi=rectangular_packet(x,t,a,k0,mass);scaled=psi/a0
    position=styled_fig('rect','위치 x / L₀','Re[Ψ]/A₀, Im[Ψ]/A₀, |Ψ|²/A₀²',540)
    for key,values,name,color,dash in [('reference',(np.abs(x)<a).astype(float),'t=0 사각형 · |Ψ|²/A₀²','#74e0aa','dash'),('real',scaled.real,'Re[Ψ] / A₀','#48d7ff',None),('imag',scaled.imag,'Im[Ψ] / A₀','#76a9ff','dot'),('density',np.abs(scaled)**2,'|Ψ|² / A₀²','#ff7daf',None)]:
        if key in opts:line(position,x,values,name,color,dash=dash,fill=key=='density')
    if 'phase' in opts:
        amp=np.abs(scaled);phase=np.angle(scaled)
        position.add_trace(go.Scatter(x=x,y=np.where(amp>.01,amp,np.nan),mode='markers',name='|Ψ| / A₀ · 색 = 위상',marker=dict(size=4,color=phase,colorscale=PHASE_COLORS,cmin=-np.pi,cmax=np.pi,colorbar=dict(title='위상',tickvals=[-np.pi,0,np.pi],ticktext=['−π','0','π'])),customdata=phase,hovertemplate='x=%{x:.3f}<br>|Ψ|/A₀=%{y:.3f}<br>위상=%{customdata:.3f} rad<extra></extra>'))
    center=k0*t/mass
    position.add_vline(x=center,line_color='white',line_dash='dash')
    position.add_annotation(x=center,xref='x',y=.98,yref='paper',text=f'대칭 중심 x<sub>c</sub><br>v<sub>c</sub> = {k0/mass:.3f} v<sub>0</sub>',showarrow=False,xanchor='right' if center>0 else 'left',xshift=-8 if center>0 else 8,bgcolor='#0d1b2e',borderpad=4)
    position.update_xaxes(range=[float(x[0]),float(x[-1])]);position.update_yaxes(range=[-1.5,1.7])
    fit_axis(position,axis,zoom)
    k=np.linspace(-12,12,1601);phi=rectangular_spectrum(k,a,k0);density=np.abs(phi)**2
    spectral=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.18,
                           subplot_titles=['초기 푸리에 진폭 φ(k,0) · 부호가 있는 sinc 함수','파수 확률밀도 |φ(k)|² · 시간에 무관'])
    spectral.update_layout(styled_fig('rect-k','','',620).layout)
    spectral.update_layout(showlegend=False,margin=dict(l=100,r=30,t=65,b=65))
    spectral.add_trace(go.Scatter(x=k,y=phi,mode='lines',name='φ(k,0)',line=dict(color='#48d7ff',width=2.8),hovertemplate='k=%{x:.3f}<br>φ(k,0)=%{y:.5f}<extra></extra>'),row=1,col=1)
    spectral.add_trace(go.Scatter(x=k,y=density,mode='lines',name='|φ(k)|²',line=dict(color='#ffd166',width=2.8),fill='tozeroy',hovertemplate='k=%{x:.3f}<br>|φ(k)|²=%{y:.5f}<extra></extra>'),row=2,col=1)
    for row in (1,2):
        spectral.add_vrect(x0=k0-np.pi/a,x1=k0+np.pi/a,fillcolor='#48d7ff',opacity=.07,line_width=0,row=row,col=1)
        for v in (k0-np.pi/a,k0+np.pi/a):spectral.add_vline(x=v,line_dash='dot',line_color='#48d7ff',row=row,col=1)
        spectral.add_vline(x=k0,line_dash='dash',line_color='white',row=row,col=1)
    spectral.update_xaxes(range=[-12,12],gridcolor='#1c3550',zerolinecolor='#496078')
    spectral.update_yaxes(gridcolor='#1c3550',zerolinecolor='#a8bad0',zerolinewidth=1.4)
    spectral.update_xaxes(title='파수 k · 단위 1/L₀',row=2,col=1)
    spectral.update_yaxes(title='초기 진폭 φ(k,0)',range=[-.35*np.sqrt(a/np.pi),1.12*np.sqrt(a/np.pi)],row=1,col=1)
    spectral.update_yaxes(title='확률밀도 |φ(k)|²',range=[0,1.12*a/np.pi],row=2,col=1)
    for annotation,color in zip(spectral.layout.annotations,['#48d7ff','#ffd166']):annotation.font=dict(size=18,color=color)
    metrics=[metric('대칭 중심 x_c',f'{center:.3f}'),metric('중심 이동 속도 v_c / v₀',f'{k0/mass:.3f}'),metric('초기 위치 표준편차 Δx₀',f'{a/np.sqrt(3):.3f}'),metric('파수 중심 봉우리 폭 · 첫 영점 사이',f'{2*np.pi/a:.3f}')]
    concepts=[math_block(r'|\phi(k)|^2=\frac{a}{\pi}\left(\frac{\sin[a(k-k_0)]}{a(k-k_0)}\right)^2',r'k_{\mathrm{zero}}=k_0\pm\frac{\pi}{a},\qquad W_k=\frac{2\pi}{a}'),
              html.P('a가 작으면 위치 분포가 좁고 파수 중심 봉우리가 넓습니다. a가 커지면 파수 분포는 좁고 높아집니다. 규격화된 전체 위치·파수 확률은 각각 1이며, 화면에 보이는 부분에서 다시 정규화하지 않습니다.'),
              math_block(r'\Psi_{k_0}(x,t)=e^{i(k_0x-k_0^2t/(2m))}\Psi_{0}\left(x-\frac{k_0t}{m},t\right)',r'|\Psi_{k_0}(x,t)|^2=\left|\Psi_0\left(x-\frac{k_0t}{m},t\right)\right|^2'),
              html.P('위 식은 ℏ=1 단위입니다. |φ(k)|²가 k₀를 중심으로 좌우 대칭이어도 k₀>0이면 위치 분포의 대칭 중심은 오른쪽으로 이동합니다. 모든 파수가 양수인 상태는 아니며 음의 k 꼬리도 남습니다. 초기 밀도만으로 이동 방향을 정할 수 없고 e^(ik₀x)의 공간 위상이 필요합니다.'),
              html.Details([html.Summary('사각형에서 폭을 표준편차로 쓰지 않는 이유'),html.P('날카로운 불연속 때문에 |φ(k)|²의 꼬리는 1/k² 크기로 감소합니다. 따라서 ∫k²|φ(k)|² dk는 발산하고 Δk와 평균 운동에너지는 유한하지 않습니다. t=0의 Δx₀=a/√3은 유한하지만 t>0의 위치 분산도 발산합니다. 이 예시는 가우스 최소 불확정성 상태가 아닙니다.'),html.P('또한 절대수렴하는 평균 파수·평균 위치를 일반적으로 사용할 수 없어, 여기서는 ⟨x⟩ 대신 분포의 대칭 중심 x_c=k₀t/m를 표시합니다. 대칭적인 주값과 통상적인 기댓값은 구분합니다. 실제로 매끈한 경계로 준비한 파동묶음에서는 이러한 발산이 완화됩니다.')]),
              html.P('길이 L₀, 질량 M₀, 시간 T₀=M₀L₀²/ℏ, 속도 v₀=ℏ/(M₀L₀)의 단위를 사용합니다. A₀=1/√(2a)는 초기 진폭 기준이며 그래프의 상대 높이는 확률 자체가 아닙니다.')]
    qa=[('a를 두 배로 하면 파수 분포는 어떻게 변하나요?','첫 영점 사이 폭 2π/a는 절반이 되고 중심 높이 a/π는 두 배가 됩니다. 전체 파수 확률은 1로 유지됩니다.'),('밀도는 같아 보이는데 k₀를 바꾸면 왜 움직이나요?','초기 |Ψ|²는 같지만 Ψ에는 e^(ik₀x)의 다른 공간 위상이 들어 있습니다. 파수 분포의 대칭 중심이 0에서 k₀로 이동하고, 위치 밀도는 k₀t/m만큼 옮겨집니다.'),('위치 분포가 퍼질 때 파수 분포도 변하나요?','자유입자에서는 φ(k,t)=φ(k,0)e^(−ik²t/(2m))이므로 |φ(k,t)|²는 일정합니다. 성분 사이의 상대 위상 변화로 위치 분포가 달라집니다.')]
    questions=[html.Li([html.P(q),html.Details([html.Summary('설명 확인'),html.P(answer)])]) for q,answer in qa]
    note='A₀=1/√(2a). 흰 점선은 평균 위치가 아닌 대칭 중심 x_c입니다. 화면 밖에도 확률 꼬리가 있으며, 고정된 표시 구간을 경계나 우물로 해석하지 않습니다.'
    knote='위: 초기 진폭 φ(k,0)는 실수 sinc 함수로 음의 곁봉우리도 있습니다. 아래: 확률밀도 |φ(k)|²는 항상 0 이상입니다. 두 그래프는 초기 진폭과 그 제곱을 비교하도록 재생 중 고정합니다. 실제 φ(k,t)는 exp[−ik²t/(2m)]의 위상을 얻으며 |φ(k,t)|²만 시간에 무관합니다. 파란 점선은 k₀±π/a의 첫 영점, 옅은 영역은 중심 봉우리입니다. 폭 W_k=2π/a는 표준편차 Δk와 다릅니다. x축과 k축의 범위는 재생 중 고정됩니다.'
    return position,spectral,metrics,concepts,questions,'',note,knote,'교과서 예시 · 사각 파동묶음의 자유 운동',dict(x=x,real=psi.real,imag=psi.imag,density=np.abs(psi)**2)
