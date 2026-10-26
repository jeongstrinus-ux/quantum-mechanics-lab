"""Same Gaussian spectrum, different dispersion laws; exact continuum solutions."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html,dcc
from teaching_content import math_block
from physics import gaussian_packet,gaussian_width,k_density

X=np.linspace(-12,36,1801)


def fields(x,t,k,sk,m):
    sx=1/(2*sk);a0=(2*np.pi*sx*sx)**(-.25)
    linear=np.exp(-(x-k*t/m)**2/(4*sx*sx))*np.exp(1j*k*(x-k*t/m))
    matter=gaussian_packet(x,t,k,sx,m)/a0
    return linear,matter


def model(k,sk,m,options,axis,zoom):
    return dict(mode='dispersion',x=X.tolist(),k=k,sk=sk,mass=m,options=options or [],axis=axis,zoom=zoom)


def equations():
    return [math_block(r'F(x,t)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}A(k)e^{i[kx-\omega(k)t]}\,dk',r'A(k)\propto e^{-(k-k_0)^2/(4\sigma_k^2)},\qquad \sigma_{x0}=\frac{1}{2\sigma_k}'),
            dcc.Markdown(r'**같은 초기 파수 분포**'+r'를 두 분산관계로 전파합니다. 위: $\omega=ck$, $c=\hbar k_0/m$. 아래: $\omega=\hbar k^2/(2m)$. 무분산 속도를 물질파의 중심 속도에 맞춰 **두 중심이 함께 이동**합니다. 두 경우 모두 유한 개수의 합이 아닌 **모든 파수의 연속 가우스 중첩**을 사용합니다.',mathjax=True)]


def figures(k,sk,m,t,options,axis,zoom,styled_fig,line,metric,fit_axis):
    sx=1/(2*sk);a0=(2*np.pi*sx*sx)**(-.25);opts=set(options or [])
    waves=fields(X,t,k,sk,m);centers=[k*t/m,k*t/m];widths=[sx,gaussian_width(t,sx,m)]
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.20,subplot_titles=['무분산 파동 · 모양을 유지하며 이동','자유입자의 물질파 · 이동하면서 퍼짐'])
    f.update_layout(styled_fig('dispersion','','',740).layout)
    for row,(wave,center,width) in enumerate(zip(waves,centers,widths),1):
        # Metadata keys let browser playback update values without relying on labels.
        traces=[]
        if 'real' in opts:traces.append(('real',wave.real,'실수부','#48d7ff',None))
        if 'density' in opts:traces.append(('density',np.abs(wave)**2,'진폭의 절댓값 제곱','#ff7daf',None))
        if 'envelope' in opts:
            traces.extend([('env',np.abs(wave),'위쪽 포락선','#ffd166','dash'),('neg-env',-np.abs(wave),'아래쪽 포락선','#ffd166','dash')])
        if 'reference' in opts:traces.append(('reference',np.exp(-(X-center)**2/(2*sx*sx)),'초기 폭 유지 · 중심만 이동한 기준','#74e0aa','dot'))
        for key,values,name,color,dash in traces:
            f.add_trace(go.Scatter(x=X,y=values,mode='lines',name=name,legendgroup=key,showlegend=row==1,meta=dict(row=row,field=key),line=dict(color=color,width=2.5,dash=dash)),row=row,col=1)
        f.add_vline(x=center,line_color='white',line_dash='dash',line_width=2,row=row,col=1)
        f.add_vrect(x0=center-width,x1=center+width,fillcolor='#48d7ff',opacity=.07,line_width=0,row=row,col=1)
    f.update_xaxes(range=[-12,36],gridcolor='#1c3550',zerolinecolor='#496078')
    f.update_yaxes(range=[-1.2/zoom,1.9/zoom],title='초기 진폭을 기준으로 한 상대 높이',gridcolor='#1c3550',zerolinecolor='#496078')
    f.update_xaxes(title='위치 x / L₀',row=2,col=1)
    f.update_layout(legend=dict(y=1.10,font=dict(size=15)),margin=dict(l=100,r=35,t=130,b=65))
    for a in f.layout.annotations:a.font=dict(size=18,color='#d2e1f0')
    for row,(center,width) in enumerate(zip(centers,widths),1):
        f.add_annotation(x=center,y=1.6/zoom,xref='x' if row==1 else 'x2',yref='y' if row==1 else 'y2',text=f'폭 σ<sub>x</sub> = {width:.3f} L₀ · 초기의 {width/sx:.2f}배',showarrow=False,xanchor='left',font=dict(size=16,color='#d2e1f0'),bgcolor='#071524')
    spectral=styled_fig('dispersion-k','파수 k · 단위 1/L₀','같은 초기 성분 분포 |A(k)|²')
    ks=np.linspace(k-5*sk,k+5*sk,801);line(spectral,ks,k_density(ks,k,sk),'두 모형 공통 · |A(k)|²','#76a9ff',fill=True)
    metrics=[metric('무분산 · 폭 / 초기 폭','1.000'),metric('물질파 · 폭 / 초기 폭',f'{widths[1]/sx:.3f}'),metric('무분산 · 중심 속도',f'{k/m:.3f} V₀'),metric('물질파 · 중심 속도',f'{k/m:.3f} V₀')]
    concepts=[math_block(r'F_{\rm linear}(x,t)=F(x-ct,0),\qquad \sigma_x(t)=\sigma_{x0}',r'\sigma_{x,\rm matter}(t)=\sigma_{x0}\sqrt{1+\left(\frac{\hbar t}{2m\sigma_{x0}^2}\right)^2}'),
        html.P('무분산에서는 모든 성분의 위상속도가 c로 같습니다. 각 성분의 초기 위상 관계가 평행이동으로 보존되어 내부 진동과 포락선 전체가 함께 이동합니다. 물질파에서는 ω(k)의 곡률 때문에 성분들의 상대 위상이 달라져 포락선이 퍼집니다. 파수 분포 |A(k)|² 자체는 두 경우 모두 시간에 따라 변하지 않습니다.'),
        html.P('흰 점선은 |F|² 분포의 중심, 밝은 음영은 중심 ±σx입니다. σx는 진폭의 절댓값 제곱을 규격화한 분포의 표준편차입니다. 물질파에서는 이것이 위치 불확정도 Δx입니다. 노란 파선은 ±|F|/A₀, 분홍선은 |F|²/A₀²이며 서로 다른 양입니다. A₀는 공통 초기 중심 진폭으로 고정하므로, 퍼질 때 봉우리 높이가 실제로 낮아집니다.'),
        html.P('무분산 파동의 F는 파동을 기술하는 복소 진폭이고, 그 실수부가 실제 진동을 나타냅니다. 이 모형의 |F|²를 입자 위치 확률로 해석하지 않습니다. 물질파의 F는 Ψ이며 |Ψ|²가 위치 확률밀도입니다. 양자역학적 확률 해석은 물질파에 적용합니다.'),
        html.P('위쪽은 오른쪽으로 진행하는 무분산 가지 ω=ck를 전체 k에 사용합니다. 일반적인 ω=c|k|의 양의 진동수 가지와 구분합니다. 동일한 초기 파형의 이동을 비교하는 모형이며 흡수·반사·비선형 효과는 없습니다. 아래쪽은 비상대론적 자유입자입니다.'),
        html.P('V₀=L₀/T₀. 물질파 비교를 위해 T₀=M₀L₀²/ℏ, ℏ=1 단위를 사용합니다. 퍼짐을 비교하기 위해 무분산 속도 c를 물질파의 중심 속도 ℏk₀/m에 맞췄습니다. 두 중심은 같은 속도로 이동합니다. 이것은 비교를 위한 선택이며 모든 무분산 파동이 물질파와 같은 속도로 진행한다는 뜻은 아닙니다. 퍼짐은 중심 이동과 별개이며, 질량을 키우면 물질파의 퍼짐과 두 중심의 이동이 모두 느려집니다. 위쪽 속도도 자동으로 맞추지만, 위쪽 파동에 입자 질량이 있다는 뜻은 아닙니다. 표시 구간 밖에도 파동은 존재하고 화면 안에서 다시 규격화하지 않습니다.')]
    qa=[('파수 성분이 많으면 언제나 퍼지나요?','아닙니다. 무분산에서는 연속적인 모든 파수를 중첩해도 각 성분의 위상속도가 같아 전체 파형이 유지됩니다. 퍼짐은 성분 개수 자체가 아니라 분산관계의 곡률과 관련됩니다.'),('물질파가 퍼지면 파수 분포도 넓어지나요?','여기서는 |A(k)|²가 일정합니다. 시간에 따른 상대 위상 변화가 위치 파형을 바꿉니다.'),('두 중심이 함께 이동하는데 왜 아래쪽만 넓어지나요?','중심 이동 속도를 같게 맞춰도 분산관계의 곡률이 다릅니다. σx/σx0로 폭의 변화를 비교할 수 있습니다. 점선 기준은 각 중심으로 이동한 초기 폭입니다. 물질파만 이 기준보다 넓어집니다.')]
    questions=[html.Li([html.P(q),html.Details([html.Summary('설명 확인'),html.P(a)])]) for q,a in qa]
    return f,spectral,metrics,concepts,questions,'','같은 초기 파형·같은 시간·같은 축·같은 중심 속도로 비교합니다. 위쪽은 모양 유지, 아래쪽은 퍼짐입니다. 폭은 |F|² 분포의 표준편차이며 음영은 경계가 아닙니다. 초기 높이를 고정합니다.','연속 가우스 분포의 모든 k를 사용합니다. 표시만 k₀±5σₖ로 제한하며 두 모형에서 시간 내내 같은 분포입니다.','여러 파수의 파동묶음 · 무분산과 물질파 동시 비교',dict(x=X,linear_real=waves[0].real*a0,linear_imag=waves[0].imag*a0,matter_real=waves[1].real*a0,matter_imag=waves[1].imag*a0)
