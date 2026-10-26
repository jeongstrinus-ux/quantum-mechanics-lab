"""Two-wave beating as a bridge from a plane wave to a wave packet."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html,dcc
from teaching_content import math_block

X=np.linspace(-20,40,1501)


def parameters(k,dk,m,kind='matter'):
    k2=k+dk
    w1,w2=(k,k2) if kind=='linear' else (k*k/(2*m),k2*k2/(2*m))
    return dict(k1=k,k2=k2,w1=w1,w2=w2,dk=dk,dw=w2-w1,kbar=(k+k2)/2,wbar=(w1+w2)/2,vg=(w2-w1)/dk,vp=(w1+w2)/(k+k2))


def frame(x,t,k,dk,m,kind='matter'):
    p=parameters(k,dk,m,kind)
    y1=np.cos(k*x-p['w1']*t);y2=np.cos(p['k2']*x-p['w2']*t)
    env=2*np.abs(np.cos((dk*x-p['dw']*t)/2))
    return y1,y2,y1+y2,env


def controls():
    return html.Div([html.Label('파동 모형 선택'),dcc.RadioItems(id='beat-kind',options=[{'label':'1. 무분산 파동 · ω=ck','value':'linear'},{'label':'2. 자유입자의 물질파 · ω=ℏk²/(2m)','value':'matter'}],value='linear',className='ft-shapes'),html.P('같은 초기 파형이 분산관계에 따라 어떻게 달라지는지 비교하세요.',className='hint')],id='beat-controls',style={'display':'none'})


def equations(kind='linear'):
    return [math_block(r'\omega=ck,\quad c=V_0' if kind=='linear' else r'\omega=\frac{\hbar k^2}{2m},\quad\hbar=1'),math_block(r'y_1=A\cos(k_1x-\omega_1t),\qquad y_2=A\cos(k_2x-\omega_2t)',r'\frac{y_1+y_2}{A}=2\cos\!\left(\frac{\Delta k\,x-\Delta\omega\,t}{2}\right)\cos(\bar{k}x-\bar{\omega}t)'),dcc.Markdown(r'$\Delta k=k_2-k_1$, $\Delta\omega=\omega_2-\omega_1$, $\bar{k}=(k_1+k_2)/2$, $\bar{\omega}=(\omega_1+\omega_2)/2$. 여기서 $y$는 파동의 실수 진폭이며, 물질파에서는 파동함수의 실수부를 나타냅니다. 포락선(envelope)은 빠른 진동의 **진폭이 변하는 외곽 모양**입니다.',mathjax=True)]


def figures(k,dk,m,t,options,axis,zoom,styled_fig,line,metric,fit_axis,kind='linear'):
    speed_unit='V₀' if kind=='linear' else 'v₀'
    p=parameters(k,dk,m,kind);y1,y2,total,env=frame(X,t,k,dk,m,kind)
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.20,subplot_titles=['두 평면파 · 서로 다른 파수와 진동수','합성파와 포락선 · 봉우리의 이동 비교'])
    f.update_layout(styled_fig('beat','','',700).layout)
    rows=[(y1,'파동 1 · y₁/A','#76a9ff',1,None),(y2,'파동 2 · y₂/A','#ff7daf',1,None),(total,'합성파 · (y₁+y₂)/A','#48d7ff',2,None)]
    if 'envelope' in (options or []):rows.extend([(env,'위쪽 포락선','#ffd166',2,'dash'),(-env,'아래쪽 포락선','#ffd166',2,'dash')])
    for y,name,color,row,dash in rows:f.add_trace(go.Scatter(x=X,y=y,mode='lines',name=name,line=dict(color=color,width=2.6,dash=dash)),row=row,col=1)
    for speed,color,label in [(p['vg'],'#ffd166','포락선 봉우리'),(p['vp'],'#74e0aa','내부 진동 · 같은 위상')]:
        f.add_vline(x=speed*t,line_dash='dash',line_width=2.5,line_color=color,row=2,col=1)
    f.update_xaxes(range=[-20,40],gridcolor='#1c3550',zerolinecolor='#496078')
    f.update_yaxes(range=[-1.3/zoom,2.4/zoom],title='개별 파동 / A',row=1,col=1)
    f.update_yaxes(range=[-2.5/zoom,3.5/zoom],title='합성 진폭 / A',row=2,col=1)
    f.update_xaxes(title='위치 x / L₀',row=2,col=1)
    f.update_layout(legend=dict(y=1.12),margin=dict(l=100,r=35,t=145,b=65))
    for a in f.layout.annotations:a.font=dict(size=18,color='#d2e1f0')
    f.add_annotation(x=p['vg']*t,y=3.1/zoom,text=f'포락선 속도 = {p["vg"]:.3f} {speed_unit}',font=dict(color='#ffd166',size=16),showarrow=False,xref='x2',yref='y2',xanchor='left')
    f.add_annotation(x=p['vp']*t,y=2.55/zoom,text=f'내부 진동의 위상속도 = {p["vp"]:.3f} {speed_unit}',font=dict(color='#74e0aa',size=16),showarrow=False,xref='x2',yref='y2',xanchor='left')
    # Both selected crests have phase zero at x=t=0; follow that same crest.
    for i,color,symbol in [(1,'#76a9ff','circle-open'),(2,'#ff7daf','diamond')]:
        speed=p[f'w{i}']/p[f'k{i}'];crest=speed*t
        f.add_trace(go.Scatter(x=[crest],y=[1.],mode='markers',name=f'마루 추적 {i}',showlegend=False,
                              marker=dict(size=16 if i==1 else 10,symbol=symbol,color=color,line=dict(width=3,color=color)),
                              hovertemplate=f'파동 {i} · 처음 x=0에 있던 마루<br>x=%{{x:.3f}} L₀<br>위상=0 · y/A=1<extra></extra>'),row=1,col=1)
        f.add_vline(x=crest,line_dash='dot',line_color=color,line_width=2.5,row=1,col=1)
        f.add_annotation(x=crest,y=(2.12 if i==1 else 1.58)/zoom,xref='x',yref='y',
                         text=f'파동 {i} · 위상속도 = {speed:.3f} {speed_unit}',showarrow=False,xanchor='left',
                         font=dict(color=color,size=16),bgcolor='#071524')
    s=styled_fig('beat-dispersion','파수 k · 단위 1/L₀','각진동수 ω · 단위 1/T₀')
    ks=np.linspace(0,8,301);line(s,ks,ks if kind=='linear' else ks**2/(2*m),'무분산 · ω=ck' if kind=='linear' else '자유입자 · ω=k²/(2m)','#76a9ff')
    s.add_trace(go.Scatter(x=[k,p['k2']],y=[p['w1'],p['w2']],mode='lines+markers',name='두 성분 · 할선 기울기 = Δω/Δk',line=dict(color='#ffd166',width=4),marker=dict(size=12)))
    metrics=[metric('포락선 속도 Δω/Δk',f'{p["vg"]:.3f} {speed_unit}'),metric('내부 진동의 위상속도 ω̄/k̄',f'{p["vp"]:.3f} {speed_unit}'),metric('인접 포락선 봉우리 간격',f'{2*np.pi/dk:.3f} L₀'),metric('두 파수 차이 Δk',f'{dk:.3f} /L₀')]
    concepts=[math_block(r'E_{\rm env}(x,t)=2A\left|\cos\!\left(\frac{\Delta k\,x-\Delta\omega\,t}{2}\right)\right|',r'\text{포락선 속도}=\frac{\Delta\omega}{\Delta k}\ \longrightarrow\ v_g=\left.\frac{d\omega}{dk}\right|_{\bar k}\quad(\Delta k\to0)',r'\text{내부 진동의 위상속도}=\frac{\bar\omega}{\bar k},\qquad v_{p,1}=\frac{\omega_1}{k_1},\quad v_{p,2}=\frac{\omega_2}{k_2}'),html.P('노란 위아래 파선'+'은 합성파를 감싸는 ±E_env/A입니다. 부호를 포함한 느린 코사인 인자는 음수가 될 수도 있지만 진폭의 크기인 포락선은 0 이상입니다. 노란 수직선은 t=0의 x=0에서 출발한 포락선 봉우리, 초록 수직선은 내부 코사인 인자의 같은 위상을 추적합니다. 초록선이 실제 합성파의 국소 극댓값을 항상 정확히 가리키는 것은 아닙니다.'),html.P('이 실험은 자유입자의 분산관계 ω=ℏk²/(2m)를 사용합니다. 포물선에서는 유한한 Δk에서도 Δω/Δk=ℏk̄/m가 정확합니다. 내부 진동의 위상속도 ω̄/k̄는 Δk가 작을 때 중심 파수의 위상속도 ω(k̄)/k̄에 가까워집니다. Δk/k̄가 작을수록 빠른 진동과 느린 포락선이 뚜렷하게 구분됩니다.'),html.P('두 평면파의 합은 전 공간에 반복되는 맥놀이입니다. 한 곳에 국소화된 규격화 가능한 파동묶음이 아니며, 포락선을 입자 하나의 위치 확률로 해석하지 않습니다. 다양한 파수를 더하는 다음 중첩 실험으로 연결하세요.'),html.P('ℏ=1. 길이 L₀, 질량 M₀, 시간 T₀=M₀L₀²/ℏ, 속도 v₀=ℏ/(M₀L₀). Δk는 두 파수의 차이로, 가우스 분포의 표준편차 σₖ와 다릅니다.')]
    if kind=='linear':
        concepts[2]=html.P('무분산 파동에서는 모든 성분이 ω=ck를 따릅니다. 여기서는 c=V₀로 고정합니다. 각 성분의 위상속도, 내부 진동의 위상속도, 포락선 속도가 모두 c입니다. 처음 겹쳤던 두 마루가 계속 함께 이동하며 합성파 전체는 y(x,t)=y(x−ct,0)로 모양을 유지합니다. 균일한 공기 중 작은 진폭의 음파를 무분산으로 근사하는 경우가 예시입니다.')
        concepts[4]=html.P('길이 L₀와 시간 T₀는 선택한 기준 단위이고 속도 단위는 V₀=L₀/T₀입니다. 무분산 모형에는 입자 질량이 필요하지 않습니다. Δk는 두 파수의 차이입니다. 흡수, 비선형 효과와 반사는 다루지 않습니다.')
    qa=[('Δk를 줄이면 포락선 봉우리 사이 간격은?','2π/Δk이므로 넓어집니다. 두 파동이 공간적으로 천천히 어긋납니다.'),('노란선과 초록선이 같은 속도로 움직이나요?','자유입자 분산관계에서는 서로 다릅니다. 가까운 두 파수에서는 군속도가 중심 파수 위상속도의 두 배에 가까워집니다.'),('진공 속 빛도 두 속도가 다른가요?','진공의 빛은 ω=ck이므로 두 속도가 모두 c입니다. 현재 그래프는 빛이 아닌 비상대론적 자유입자의 분산관계입니다.')]
    if kind=='linear':
        qa[1]=('노란선과 초록선이 같은 속도로 움직이나요?','두 속도가 모두 c입니다. 두 개별 파동의 마루도 함께 이동하고 합성파 전체 모양이 유지됩니다.')
        qa[2]=('물질파로 바꾸면 무엇이 달라지나요?','동일한 t=0 파형에서 시작하지만 각 성분의 위상속도가 달라져 마루가 벌어집니다. 포락선과 내부 위상도 다른 속도로 진행합니다. 두 성분만으로는 포락선 폭이 계속 퍼지지는 않습니다.')
    questions=[html.Li([html.P(q),html.Details([html.Summary('설명 확인'),html.P(a)])]) for q,a in qa]
    note='무분산 모형에서는 두 마루와 포락선이 모두 같은 속도로 움직이고 전체 파형이 유지됩니다.' if kind=='linear' else '자유입자 모형에서는 두 마루가 서로 다른 속도로 움직입니다. 포락선과 내부 위상의 속도도 다르지만, 두 성분만의 포락선은 폭이 계속 퍼지는 국소화 파동묶음이 아닙니다.'
    return f,s,metrics,concepts,questions,'',note+' 위쪽 원과 마름모는 t=0에 x=0에서 겹쳤던 두 마루(위상 0)를 각각 추적합니다. 각 마루는 x=ω₁t/k₁, x=ω₂t/k₂로 이동하므로 위상속도가 다르면 점점 떨어집니다. 마루는 각 파동의 최대 변위 y/A=1을 유지합니다. 아래쪽 노란 수직선은 포락선 봉우리의 이동을, 초록 수직선은 내부의 빠른 진동에서 같은 위상의 이동을 나타냅니다. 두 파수가 가까울 때 포락선 속도는 군속도로 연결됩니다. 자유입자 모형에서는 두 파수가 가까우면 포락선이 내부 위상보다 약 두 배 빠르게 이동합니다. 처음에는 Δk를 작게 설정하고 비교하세요.','노란 두 점은 (k₁,ω₁), (k₂,ω₂)입니다. 두 점을 잇는 직선의 기울기가 포락선의 이동 속도입니다.','두 평면파의 맥놀이 · 포락선과 군속도',dict(x=X,wave1=y1,wave2=y2,sum=total,envelope=env)
