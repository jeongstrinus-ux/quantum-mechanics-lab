"""Controls and figures for coherent superpositions of oscillator eigenstates."""
from functools import lru_cache
from math import factorial
import numpy as np
import plotly.graph_objects as go
from dash import html,dcc,Patch
from physics import oscillator_basis,oscillator_coefficients,oscillator_moments
from teaching_content import question_items,math_block

MAX_N=12
X=np.linspace(-8,8,2401)
BASIS=oscillator_basis(X,MAX_N)
PHASE_COLORS=[[0,'#ff5555'],[1/6,'#ffe35a'],[2/6,'#61e580'],[.5,'#48d7ff'],[4/6,'#7676ff'],[5/6,'#f071d5'],[1,'#ff5555']]
DEFAULT_STATE=dict(weights=[1.,1.]+[0.]*(MAX_N-1),phases=[0.]*(MAX_N+1))


def preset_values(preset):
    w=np.zeros(MAX_N+1);angles=np.zeros(MAX_N+1)
    if preset=='ground':w[0]=1
    elif preset=='adjacent':w[:2]=1
    elif preset=='breathing':w[[0,2]]=1
    elif preset=='coherent':
        alpha=1.5
        w=np.array([np.exp(-alpha**2)*alpha**(2*n)/factorial(n) for n in range(MAX_N+1)])
    else:raise ValueError('unknown oscillator preset')
    return w.tolist()+angles.tolist()


def number_control(id_,value,nonnegative=False):
    return html.Div([
        html.Button('−',id=id_+'-minus',type='button',className='ho-number-minus',**{'aria-label':'Decrease value','aria-controls':id_}),
        dcc.Input(id=id_,type='number',step='any',min=0 if nonnegative else None,max=None,value=value),
        html.Button('+',id=id_+'-plus',type='button',className='ho-number-plus',**{'aria-label':'Increase value','aria-controls':id_})
    ],className='ho-number-control')


def controls():
    rows=[]
    for n in range(MAX_N+1):
        rows.append(html.Tr([html.Td(str(n)),html.Td(number_control(f'ho-weight-{n}',DEFAULT_STATE['weights'][n],True)),
                            html.Td(number_control(f'ho-phase-{n}',0))]))
    return html.Div([
        html.H3('에너지 고유상태 중첩의 초기 조건'),
        html.Label('예시 불러오기'),dcc.Dropdown(id='ho-preset',options=[{'label':label,'value':value} for value,label in [
            ('custom','직접 입력'),('ground','바닥상태 n=0'),('adjacent','n=0과 1 · 중심 진동'),('breathing','n=0과 2 · 폭 변화'),('coherent','코히런트 상태 · 거의 일정한 폭')]],value='custom',clearable=False),
        html.P('n=0이 바닥상태입니다. wₙ은 상대 확률 가중치, θₙ은 초기 위상(도)입니다. 0이면 제외하고, 가중치 합이 1이 되도록 자동 환산합니다.',className='hint'),
        html.Div(html.Table([html.Thead(html.Tr([html.Th('n'),html.Th('wₙ'),html.Th('θₙ (°)')])),html.Tbody(rows)],className='ho-table'),className='ho-table-scroll'),
        html.P('± 버튼은 가중치 0.1, 위상 15°씩 조절합니다. 소수 값은 직접 입력할 수 있습니다. 계수 cₙ=√(wₙ/Σw) exp(iθₙ).',className='hint'),
        dcc.Store(id='ho-state',data=DEFAULT_STATE)],id='ho-controls',style={'display':'none'})


@lru_cache(maxsize=32)
def initial_data(weights,angles):
    c=oscillator_coefficients(weights,angles)
    ref=c@BASIS
    amp0=float(np.max(np.abs(ref)))
    upper=np.max(np.abs(c)@np.abs(BASIS))/amp0
    return c,ref,amp0,float(max(1.25,upper*1.1,upper**2*1.1))


def display_values(psi,ref,amp0,options):
    opts=set(options or [])
    result=[]
    for key,y in [('real',psi.real/amp0),('imag',psi.imag/amp0),('density',np.abs(psi)**2/amp0**2),('reference',np.abs(ref)**2/amp0**2)]:
        if key in opts:result.append((key,y))
    if 'phase' in opts:
        a=np.abs(psi)/amp0
        result.append(('phase',np.where(a>.01,a,np.nan)))
    return result


def energy_axis_limits(energy,values,initial_bound,axis_mode,zoom):
    low=-initial_bound if any(key in ('real','imag') for key,_ in values) else 0.
    high=initial_bound
    if axis_mode=='auto' and values:
        finite=np.concatenate([y for _,y in values]);finite=finite[np.isfinite(finite)]
        if finite.size:low=min(0.,float(finite.min()));high=max(0.,float(finite.max()))
    lo=min(-.25,energy+low-.15)
    hi=max(4.,energy+high+max(.3,.1*high))
    return [energy+(lo-energy)/zoom,energy+(hi-energy)/zoom]


def figures(state,t,options,axis_mode,zoom,styled_fig,line,metric,fit_axis):
    opts=set(options or [])
    position=styled_fig('oscillator','위치 x / ℓ','에너지 / (ℏω) · 파형 기준선 ⟨E⟩',540)
    spectrum=styled_fig('oscillator-energy','에너지 Eₙ / (ℏω)','성분 확률 |cₙ|²',360)
    concepts=[math_block('V(x)=\\frac12m\\omega^2x^2,\\quad E_n=\\hbar\\omega\\left(n+\\frac12\\right)','\\Psi(x,t)=\\sum_n c_n \\psi_n(x)e^{-iE_nt/\\hbar}'),
              dcc.Markdown('ψₙ는 $\\hat H \\psi_n=E_n\\psi_n$을 만족하는 정규화된 조화진동자의 에너지 고유함수입니다. 시간에 무관한 퍼텐셜에서 $\\psi_n(x)e^{-iE_nt/\\hbar}$로 진화하는 에너지 고유상태는 확률밀도가 일정하여 정상상태(stationary state)라고 합니다. 정상은 입자가 고전적으로 멈춰 있다는 뜻이 아닙니다. wₙ을 확률 가중치로 입력하고 $c_n=\\sqrt{\\frac{w_n}{\\sum_jw_j}}e^{i\\theta_n}$로 계산합니다. 파동함수의 중첩이며 확률을 따로 더하는 통계적 혼합과 다릅니다.',mathjax=True),
              html.P('이 탭은 ℏ=m=ω=1의 진동자 단위를 사용합니다. 실제 길이 단위 ℓ=√[ℏ/(mω)], 시간 단위는 1/ω, 운동량 단위는 √(mℏω)입니다. 다른 탭의 질량·파수 입력은 사용하지 않습니다.'),
              dcc.Markdown('$A_0=\\max_x|\\Psi(x,0)|$: 초기 최대 진폭을 고정 기준으로 사용합니다. 색은 전체 파동함수의 위상입니다. 세로 점선은 평균 위치 ⟨x⟩, 가로 점선은 평균 에너지 ⟨E⟩입니다. 퍼텐셜과 가로 기준선은 같은 에너지 축으로 읽습니다.',mathjax=True),
              math_block('\\frac{\\langle E\\rangle}{\\hbar\\omega}=\\sum_n\\frac{w_n}{\\sum_jw_j}\\left(n+\\frac12\\right)','y=\\frac{\\langle E\\rangle}{\\hbar\\omega}+f(x,t)'),html.P('평균 에너지는 재생 중 일정합니다. f는 초기 진폭 기준으로 나눈 실수부·허수부·진폭 또는 확률밀도의 표시 높이이며 에너지에 대한 편차가 아닙니다.'),
              html.P('에너지 고유상태 하나는 확률밀도가 변하지 않습니다. 서로 다른 에너지의 고유상태를 중첩하면 일반적으로 정상상태가 아닙니다. 상대 위상이 변해 확률밀도의 간섭항이 달라지고 분포가 움직이거나 모양이 바뀝니다. ⟨x⟩는 항상 진동자 운동방정식을 따르지만, 모든 중첩이 모양을 유지하는 파동묶음은 아닙니다.'),
              html.P('코히런트 예시는 α=1.5의 n=0~12 절단 근사입니다. 이상적인 코히런트 상태는 무한 개 상태의 중첩이며, 가우스 폭을 유지합니다. 이 예시의 생략 확률은 약 7.6×10⁻⁷입니다.'),
              html.P([html.A('모형 근거: MIT · Harmonic oscillator and coherent states',href='https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/79a8091ef4f18d8e3ff76a097e8db33c_MIT8_05F13_Chap_06.pdf',target='_blank')],className='mini')]
    try:
        weights=tuple(float(v) if v is not None else 0. for v in state['weights'])
        angles=tuple(float(v) if v is not None else 0. for v in state['phases'])
        if len(weights)!=MAX_N+1:raise ValueError('상태 입력의 개수를 확인하세요.')
        c,ref,amp0,ylim=initial_data(weights,angles)
    except (ValueError,TypeError,KeyError) as error:
        position.update_xaxes(range=[-8,8]);position.update_yaxes(range=[0,1.25])
        spectrum.update_xaxes(range=[0,13]);spectrum.update_yaxes(range=[0,1])
        return position,spectrum,[metric('초기 상태','입력 필요')],concepts,question_items('oscillator',0),str(error),'가중치를 하나 이상 양수로 입력하세요.','', '조화진동자 · 초기 상태를 준비하세요',dict(x=X,real=np.zeros_like(X),imag=np.zeros_like(X),density=np.zeros_like(X))
    psi=(c*np.exp(-1j*(np.arange(MAX_N+1)+.5)*t))@BASIS
    moments=oscillator_moments(t,c)
    energy=moments['energy']
    values=display_values(psi,ref,amp0,opts)
    for key,y in values:
        if key=='phase':
            amplitude=np.abs(psi)/amp0
            position.add_trace(go.Scatter(x=X,y=energy+y,mode='markers',name='|Ψ| / A₀ · 색 = 위상',
                marker=dict(size=4,color=np.angle(psi),colorscale=PHASE_COLORS,cmin=-np.pi,cmax=np.pi,
                    colorbar=dict(title='위상',tickvals=[-np.pi,0,np.pi],ticktext=['−π','0','π'],thickness=10,x=1.04)),
                customdata=np.column_stack([amplitude,np.angle(psi)]),hovertemplate='x=%{x:.3f}<br>|Ψ|/A₀=%{customdata[0]:.3f}<br>위상=%{customdata[1]:.3f} rad<extra></extra>'))
        elif key=='density':
            # Close the area at <E>, rather than filling down to the energy zero.
            position.add_trace(go.Scatter(x=np.concatenate([X,X[::-1]]),y=np.concatenate([energy+y,np.full_like(X,energy)]),
                mode='lines',fill='toself',name='|Ψ|² / A₀²',line=dict(color='#ff7daf',width=2.8),
                customdata=np.concatenate([y,np.zeros_like(y)]),hovertemplate='|Ψ|²/A₀²=%{customdata:.4f}<extra></extra>'))
        else:
            names={'real':'Re[Ψ] / A₀','imag':'Im[Ψ] / A₀','reference':'t=0 확률밀도'}
            colors={'real':'#48d7ff','imag':'#76a9ff','reference':'#74e0aa'}
            line(position,X,energy+y,names[key],colors[key],dash='dot' if key=='imag' else 'dash' if key=='reference' else None)
            position.data[-1].customdata=y
            position.data[-1].hovertemplate='%{customdata:.4f}<extra>'+names[key]+'</extra>'
    if 'potential' in opts:
        position.add_trace(go.Scatter(x=X,y=X**2/2,name='퍼텐셜 V · 에너지 축',mode='lines',line=dict(color='#a8bad0',dash='dash',width=1.5),hovertemplate='V=%{y:.3f} ℏω<extra></extra>'))
    position.add_hline(y=energy,line_color='#ffd166',line_dash='dash',line_width=1.5)
    position.add_vline(x=moments['mean_x'],line_color='white',line_dash='dash')
    position.add_annotation(x=moments['mean_x'],y=.98,xref='x',yref='paper',text='⟨x⟩',showarrow=False,xanchor='left',xshift=8)
    position.add_annotation(x=-7.8,y=energy,text=f'⟨E⟩ = {energy:.3f} ℏω · 파형 0',showarrow=False,xanchor='left',yanchor='bottom',yshift=4,font=dict(color='#ffd166',size=16),bgcolor='#071524')
    position.update_layout(margin=dict(l=70,r=75,t=70,b=55),yaxis_uirevision=f'ho-energy-{energy}-{axis_mode}-{zoom}')
    position.update_xaxes(range=[-8,8])
    position.update_yaxes(range=energy_axis_limits(energy,values,ylim,axis_mode,zoom),autorange=False,nticks=7)
    n=np.arange(MAX_N+1);p=np.abs(c)**2
    spectrum.add_trace(go.Bar(x=n+.5,y=p,name='고정 성분 확률',marker_color='#ffd166',customdata=np.column_stack([n,np.array(angles)]),
        hovertemplate='n=%{customdata[0]:.0f}<br>Eₙ=%{x:.1f} ℏω<br>|cₙ|²=%{y:.5f}<br>초기 위상=%{customdata[1]:.1f}°<extra></extra>'))
    spectrum.update_xaxes(range=[0,13],tickvals=n+.5,ticktext=[f'{v+.5:g}' for v in n])
    spectrum.update_yaxes(range=[0,1.05])
    spectrum.add_vline(x=energy,line_color='white',line_dash='dash')
    spectrum.add_annotation(x=energy,y=1.02,xref='x',yref='paper',text='⟨E⟩',showarrow=False,yanchor='bottom')
    metrics=[metric('평균 위치 ⟨x⟩',f"{moments['mean_x']:.3f}"),metric('위치 표준편차 Δx',f"{moments['sd_x']:.3f}"),
             metric('현재 Δx·Δp / ℏ',f"{moments['sd_x']*moments['sd_p']:.5f}"),metric('평균 에너지 / ℏω',f"{moments['energy']:.3f}")]
    note='고정 x축 −8~8. 전체 공간에서 정규화된 에너지 고유상태를 중첩하며 화면 구간에서 다시 정규화하지 않습니다. 가로 기준선은 ⟨E⟩입니다. 파형의 위아래 거리는 정규화한 파동함수·확률밀도의 표시 높이이며 에너지 차이가 아닙니다. 기준선이 에너지 고유값 하나를 의미하는 것도 아닙니다. 초기 기준 y축 범위는 재생 중 유지합니다.'
    knote='가중치를 정규화한 |cₙ|²입니다. 시간에 무관하며 에너지 측정에서 Eₙ이 나올 확률입니다. 막대는 연속 확률밀도가 아닌 이산 확률입니다.'
    return position,spectrum,metrics,concepts,question_items('oscillator',0),'',note,knote,'조화진동자 · 에너지 고유상태 중첩의 시간 변화',dict(x=X,real=psi.real,imag=psi.imag,density=np.abs(psi)**2)


def time_patch(state,t,options,axis_mode,zoom,metric):
    weights=tuple(float(v) if v is not None else 0. for v in state['weights'])
    angles=tuple(float(v) if v is not None else 0. for v in state['phases'])
    c,ref,amp0,_=initial_data(weights,angles)
    psi=(c*np.exp(-1j*(np.arange(MAX_N+1)+.5)*t))@BASIS
    moments=oscillator_moments(t,c);opts=set(options or [])
    patch=Patch();energy=moments['energy']
    values=display_values(psi,ref,amp0,opts)
    for index,(key,y) in enumerate(values):
        if key=='phase':
            phase=np.angle(psi).tolist()
            patch['data'][index]['y']=(energy+y).tolist()
            patch['data'][index]['marker']['color']=phase
            patch['data'][index]['customdata']=np.column_stack([np.abs(psi)/amp0,phase]).tolist()
        elif key=='density':
            patch['data'][index]['y']=np.concatenate([energy+y,np.full_like(X,energy)]).tolist()
            patch['data'][index]['customdata']=np.concatenate([y,np.zeros_like(y)]).tolist()
        elif key!='reference':
            patch['data'][index]['y']=(energy+y).tolist()
            patch['data'][index]['customdata']=y.tolist()
    patch['layout']['shapes'][1]['x0']=moments['mean_x']
    patch['layout']['shapes'][1]['x1']=moments['mean_x']
    patch['layout']['annotations'][0]['x']=moments['mean_x']
    if axis_mode=='auto':
        patch['layout']['yaxis']['range']=energy_axis_limits(energy,values,1,axis_mode,zoom)
    metrics=[metric('평균 위치 ⟨x⟩',f"{moments['mean_x']:.3f}"),metric('위치 표준편차 Δx',f"{moments['sd_x']:.3f}"),
             metric('현재 Δx·Δp / ℏ',f"{moments['sd_x']*moments['sd_p']:.5f}"),metric('평균 에너지 / ℏω',f"{moments['energy']:.3f}")]
    return patch,metrics
