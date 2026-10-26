from __future__ import annotations
import argparse
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, Patch, no_update, ctx, dcc, html
from plotly.subplots import make_subplots
from teaching_content import question_items,math_block
import harmonic_ui as ho
import fourier_ui as ft
import well_ui as well
import browser_playback as playback
import rectangular_ui as rect
import beat_ui as beat
import dispersion_ui as dispersion
from physics import gaussian_packet, gaussian_width, k_density, progressive_superposition, mirror_fields, mirror_envelopes, mirror_component, wall_packet, wall_plane_mode

CYAN, PINK, YELLOW, BLUE, GREEN = '#48d7ff', '#ff7daf', '#ffd166', '#76a9ff', '#74e0aa'
COMPONENT_COLORS = ['#48d7ff', '#ffad66', '#ff7daf', '#74e0aa', '#ba9cff']
# NIST/CODATA 2022: reference examples, not changes to dimensionless dynamics.
HBAR_SI=6.62607015e-34/(2*np.pi)
ELECTRON_MASS_SI=9.1093837139e-31


def electron_units(length_m):
    return HBAR_SI/(ELECTRON_MASS_SI*length_m),ELECTRON_MASS_SI*length_m**2/HBAR_SI


def free_viewport(k0,mass,sigma_k):
    velocity=k0/mass
    right=max(100.,30.*velocity)
    left=-max(15.,6/(2*sigma_k))
    return left,right,right/velocity


def representative_indices(count):
    return np.unique(np.linspace(0,count-1,min(5,count)).astype(int))


def reflection_wavenumbers(k0,pulse_sigma):
    spread=min(1/pulse_sigma,.8*k0)
    return np.linspace(k0-spread,k0+spread,5)


def component_circles(fig,ks,heights,colors,label,reflected=False):
    fig.add_trace(go.Scatter(x=ks,y=heights,mode='markers',name=label,
                            marker=dict(symbol='circle-open' if reflected else 'circle',size=13,
                                        color=colors,line=dict(color=colors if reflected else '#07111f',width=2.5)),
                            customdata=np.arange(1,len(ks)+1),
                            hovertemplate='대표 성분 %{customdata}<br>k=%{x:.3f}<br>분포 높이=%{y:.4f}<extra></extra>'))
MODE_INFO = {
    'dispersion': ('02','여러 파수의 파동묶음 · 무분산과 물질파','같은 초기 가우스 파동묶음이 모양을 유지하거나 퍼지는 모습을 비교합니다.'),
    'beat': ('01-2','맥놀이와 군속도 · 두 분산관계 비교','두 평면파가 만드는 포락선과 내부 위상의 이동을 비교합니다.'),
    'rect': ('04-2','자유입자 · 사각 파동묶음','사각형 초기 상태의 푸리에 변환과 폭, 공간 위상에 따른 자유 운동을 비교합니다.'),
    'well': ('01', '무한 퍼텐셜 우물 · 에너지 고유상태와 중첩', '양쪽 벽의 경계조건, 에너지 양자화와 두 에너지 고유상태의 중첩에 따른 간섭을 탐구합니다.'),
    'plane': ('01', '자유입자 · 평면파', '하나의 파수는 운동량을 정하지만 위치를 국소화하지 않습니다.'),
    'build': ('02', '자유입자 · 평면파의 중첩과 불확정성', '파수 폭을 바꾸고, 위치 분포의 폭이 어떻게 달라지는지 비교합니다.'),
    'free': ('03', '자유입자 · 파동묶음의 운동', '파동묶음의 중심 이동과 폭 증가를 따로 관찰합니다.'),
    'fourier': ('06', '자유입자 · 푸리에 변환', '직교성으로 성분을 추출하고 부분합으로 재구성한 뒤 연속 변환으로 연결합니다.'),
    'oscillator': ('05', '조화진동자 · 에너지 고유상태 중첩', '에너지 고유상태의 비율과 상대 위상을 정하고 시간에 따른 운동과 모양 변화를 관찰합니다.'),
    'wall': ('04', '자유입자 · 무한 퍼텐셜 벽 반사', '퍼지는 입자파가 무한 퍼텐셜 벽에서 반사되고 운동량 방향이 바뀝니다.'),
}


@lru_cache(maxsize=32)
def wall_end_time(k0,sk,mass):
    """End when the actual half-line mean reaches -40 on the outgoing leg."""
    sx=1/(2*sk)
    upper=58/k0
    length=max(100.,18+10*gaussian_width(upper,sx,1))
    n=2**int(np.ceil(np.log2(max(8192,length*4*(k0+8*sk)/np.pi))))
    dx=length/n
    x=-length+np.arange(n)*dx
    def mean(t):
        return float(np.sum(x*wall_packet(x,t,k0,sx,1)['density'])*dx)
    while mean(upper)>-40:
        upper*=1.5
        length=max(100.,18+10*gaussian_width(upper,sx,1))
        n=2**int(np.ceil(np.log2(max(8192,length*4*(k0+8*sk)/np.pi))))
        dx=length/n
        x=-length+np.arange(n)*dx
    lower=0.
    for _ in range(25):
        mid=(lower+upper)/2
        if mean(mid)>-40:lower=mid
        else:upper=mid
    return mass*upper


@lru_cache(maxsize=8)
def wall_frame(k0,sk,mass,t):
    sx=1/(2*sk)
    end=wall_end_time(k0,sk,mass)
    width_end=gaussian_width(end,sx,mass)
    x=np.linspace(-60,3,1801)
    fields=wall_packet(x,t,k0,sx,mass)
    amp0=abs(wall_packet([-18],0,k0,sx,mass)['psi'][0])
    length=max(80.,18+10*width_end)
    # Resolve the initial narrow packet even when the final packet is very broad.
    n=2**int(np.ceil(np.log2(max(8192,length*4*(k0+8*sk)/np.pi))))
    dx=length/n
    xf=-length+np.arange(n)*dx
    psi=wall_packet(xf,t,k0,sx,mass)['psi']
    rho=np.abs(psi)**2
    norm=float(np.sum(rho)*dx)
    mean=float(np.sum(xf*rho)*dx)
    sd=float(np.sqrt(max(0,np.sum((xf-mean)**2*rho)*dx)))
    k=np.fft.fftshift(2*np.pi*np.fft.fftfreq(n,dx))
    prob=np.abs(np.fft.fftshift(np.fft.fft(psi))*dx/np.sqrt(2*np.pi))**2
    dk=k[1]-k[0]
    mean_k=float(np.sum(k*prob)*dk)
    span=max(8.,k0+5*sk)
    keep=np.abs(k)<=span
    return dict(x=x,fields=fields,amp0=amp0,k=k[keep],prob=prob[keep],mean=mean,sd=sd,norm=norm,mean_k=mean_k,end=end,spectral_norm=float(np.sum(prob)*dk),k0=k0,sk=sk,mass=mass,t=t)


def wall_values(frame,options):
    fields=frame['fields'];a=frame['amp0'];values=[]
    if 'real' in options:values.append(fields['psi'].real/a)
    if 'imag' in options:values.append(fields['psi'].imag/a)
    if 'density' in options:values.append(fields['density']/a**2)
    if 'components' in options:values.extend([fields['incident'].real/a,fields['reflected'].real/a])
    if 'phase' in options:values.append(wall_phase_values(frame))
    return values


def wall_phase_values(frame):
    amplitude=np.abs(frame['fields']['psi'])/frame['amp0']
    return np.where((frame['x']<0)&(amplitude>0.01),amplitude,np.nan)


def wall_metrics(frame):
    return [metric('평균 위치 ⟨x⟩',f"{frame['mean']:.3f}"),metric('위치 표준편차 Δx',f"{frame['sd']:.3f}"),
            metric('평균 파수 ⟨k⟩',f"{frame['mean_k']:.3f}"),metric('전체 위치 확률',f"{frame['norm']:.6f}")]


def wall_center_annotation(mean):
    return dict(x=mean,y=.97,xref='x',yref='paper',text='⟨x⟩',showarrow=False,xanchor='left',xshift=8,yanchor='top',font=dict(color='white',size=17),bgcolor='#071524')


def wall_frame_patches(k0,sk,mass,t,options,axis_mode,zoom):
    frame=wall_frame(k0,sk,mass,t)
    position=Patch();spectral=Patch()
    values=wall_values(frame,options)
    stride=max(1,int(np.ceil(len(frame['x'])/480)))
    for i,y in enumerate(values):
        if 'phase' in options and i==len(values)-1:y=y[::stride]
        position['data'][i]['y']=y.tolist()
    if 'phase' in options:
        position['data'][len(values)-1]['marker']['color']=np.angle(frame['fields']['psi'])[::stride].tolist()
        position['data'][len(values)-1]['customdata']=np.angle(frame['fields']['psi'])[::stride].tolist()
    position['layout']['annotations']=[wall_center_annotation(frame['mean'])]
    position['layout']['shapes'][2]['x0']=frame['mean']
    position['layout']['shapes'][2]['x1']=frame['mean']
    if axis_mode=='auto' and values:
        low=min(0.,min(float(np.nanmin(y)) for y in values));high=max(0.,max(float(np.nanmax(y)) for y in values))
        edge=max(abs(low),abs(high),1e-8)*1.12/zoom
        position['layout']['yaxis']['range']=[-edge,edge] if low < -1e-9 else [-.035*edge,edge]
    spectral['data'][0]['y']=frame['prob'].tolist()
    return position,spectral,wall_metrics(frame)


def slider(id_, name, lo, hi, step, value, hint=''):
    return html.Div([html.Div([html.Label(name, htmlFor=id_,id=id_+'-label'), html.Output(id=id_+'-value', className='value')], className='labelrow'),
                     dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=None),
                     html.P(hint,className='hint',id=id_+'-hint')], id=id_+'-box',className='control')


MIRROR_X=np.linspace(-30,5,801)
MIRROR_COMPONENT_X=np.linspace(-30,0,601)


def mirror_direction_labels(k0,sigma,t):
    """Direction guide at the geometric pulse center, not extra electric fields."""
    reflected=t>=18
    center=-abs(t-18)
    labels=[]
    for i,(k,color) in enumerate(zip(reflection_wavenumbers(k0,sigma),COMPONENT_COLORS)):
        labels.append(dict(x=center,xref='x',y=.94-.065*i,yref='paper',
            text=f'← −k = {k:.2f}' if reflected else f'+k = {k:.2f} →',
            showarrow=False,xanchor='right' if reflected else 'left',
            xshift=-8 if reflected else 8,yanchor='top',align='left',
            font=dict(size=16,color=color),bgcolor='#071524',borderpad=2))
    return labels



def mirror_overlay_component(t,k,k0,sigma):
    """Fourier sample weighted by its k-bin width, in the pulse's E0 units."""
    ks=reflection_wavenumbers(k0,sigma)
    dk=ks[1]-ks[0]
    center_weight=sigma/np.sqrt(np.pi)*(1+np.exp(-4*sigma*sigma*k0*k0))
    p=mirror_component(MIRROR_COMPONENT_X,t,float(k),sigma,k0)
    return {key:p[key]*center_weight*dk for key in ('incident','reflected','electric')}


def mirror_position_values(k0,sigma,t,options):
    """Values in the stable trace order used by the static reflection figure."""
    options=set(options or [])
    fields=mirror_fields(MIRROR_X,t,k0,sigma)
    values=[]
    if 'components' in options:values.extend([fields['incident'],fields['reflected']])
    if 'real' in options:values.append(fields['electric'])
    if 'imag' in options:values.append(fields['magnetic'])
    if 'density' in options:values.append(fields['energy']/2)
    if 'envelope' in options:values.extend(mirror_envelopes(MIRROR_X,t,sigma))
    for i,k in enumerate(reflection_wavenumbers(k0,sigma)):
        if f'k{i+1}' in options:
            p=mirror_overlay_component(t,k,k0,sigma)
            values.extend(p[key] for key in ('incident','reflected','electric'))
    return values


def mirror_position_patch(k0,sigma,t,options,axis_mode,zoom):
    values=mirror_position_values(k0,sigma,t,options)
    patch=Patch()
    for i,y in enumerate(values):patch['data'][i]['y']=y.tolist()
    if 'directions' in (options or []):
        patch['layout']['annotations']=[dict(x=5,y=1,xref='x',yref='paper',text='이상적 반사체',showarrow=False,xanchor='right',yanchor='bottom')]+mirror_direction_labels(k0,sigma,t)
    if axis_mode=='auto' and values:
        low=min(0.,min(float(np.nanmin(y)) for y in values))
        high=max(0.,max(float(np.nanmax(y)) for y in values))
        edge=max(abs(low),abs(high),1e-8)*1.12/zoom
        patch['layout']['yaxis']['range']=[-edge,edge] if low < -1e-9 else [-.035*edge,edge]
    return patch


def mirror_components_patch(k0,sigma,t):
    patch=Patch()
    for row,k in enumerate(reflection_wavenumbers(k0,sigma)):
        p=mirror_component(MIRROR_COMPONENT_X,t,float(k),sigma,k0)
        for col,key in enumerate(('incident','reflected','electric')):
            patch['data'][3*row+col]['y']=p[key].tolist()
    return patch


def metric(name, value):
    return html.Div([html.Small(name), html.Strong(value)],className='metric')


def styled_fig(title, x_title, y_title, height=360):
    fig=go.Figure()
    fig.update_layout(template='plotly_dark',height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#071524',
                      margin=dict(l=100,r=30,t=100,b=70),font=dict(family='Arial, sans-serif',color='#d2e1f0',size=17),
                      legend=dict(orientation='h',y=1.03,x=0,yanchor='bottom',font=dict(size=16)),
                      xaxis=dict(title=x_title,gridcolor='#1c3550',zerolinecolor='#496078'),
                      yaxis=dict(title=y_title,gridcolor='#1c3550',zerolinecolor='#496078'),
                      hoverlabel=dict(bgcolor='#10223a',bordercolor='#496078',font=dict(color='#f4f8ff',size=16)),
                      hovermode='x unified',uirevision=title)
    return fig


def line(fig,x,y,name,color,dash=None,width=2.8,fill=False):
    fig.add_trace(go.Scatter(x=x,y=y,name=name,mode='lines',line=dict(color=color,width=width,dash=dash),
                            fill='tozeroy' if fill else None,hovertemplate='%{y:.4f}<extra>'+name+'</extra>'))


def fit_vertical_axis(fig, axis_mode='auto', zoom=1.0):
    """Change viewport only: never rescale values or probability normalization."""
    values=[np.asarray(trace.y,dtype=float) for trace in fig.data if trace.visible not in (False,'legendonly') and trace.y is not None and trace.yaxis in (None,'y')]
    if not values:
        return
    vals=np.concatenate(values)
    vals=vals[np.isfinite(vals)]
    if not vals.size:
        return
    if axis_mode=='auto':
        low=min(0.,float(vals.min()));high=max(0.,float(vals.max()))
        if low < -1e-9:
            edge=max(abs(low),abs(high),1e-8)*1.12/zoom
            limits=[-edge,edge]
        else:
            edge=max(high,1e-8)*1.12/zoom
            limits=[-.035*edge,edge]
    else:
        low,high=fig.layout.yaxis.range or (-1.15,1.25)
        limits=[low/zoom,high/zoom]
    fig.update_yaxes(range=limits,autorange=False,nticks=7,selector=dict(overlaying=None))
    fig.update_layout(yaxis_uirevision=f'{axis_mode}-{zoom}')


def figures(mode,k0,sigma_k,count,mass,pulse_sigma,t,options,axis_mode='fixed',zoom=1.0,harmonic_state=None,fourier_state=None,well_state=None,rect_state=None,beat_kind='linear'):
    """All plots and metrics derive from physics.py; no animation-only motion."""
    if mode=='dispersion':return dispersion.figures(k0,sigma_k,mass,t,options,axis_mode,zoom,styled_fig,line,metric,fit_vertical_axis)
    if mode=='beat':return beat.figures(k0,sigma_k,mass,t,options,axis_mode,zoom,styled_fig,line,metric,fit_vertical_axis,beat_kind)
    if mode=='rect':
        return rect.figures(rect_state or rect.DEFAULT_STATE,mass,t,options,axis_mode,zoom,styled_fig,line,metric,fit_vertical_axis)
    if mode=='well':
        return well.figures(well_state or well.DEFAULT_STATE,t,options,axis_mode,zoom,styled_fig,line,metric,fit_vertical_axis)
    if mode=='fourier':
        return ft.figures(fourier_state or ft.DEFAULT_STATE,axis_mode,zoom,styled_fig,line,metric,fit_vertical_axis)
    if mode=='oscillator':
        return ho.figures(harmonic_state or ho.DEFAULT_STATE,t,options,axis_mode,zoom,styled_fig,line,metric,fit_vertical_axis)
    opts=set(options or [])
    position=styled_fig(mode,'위치 x','상대 높이',540)
    spectral=styled_fig(mode+'-k','파수 k','파수 확률밀도 |A(k)|²',360)
    sx0=1/(2*sigma_k)
    concepts=[];warning=''
    if mode=='wall':
        frame=wall_frame(k0,sigma_k,mass,t)
        x=frame['x'];f=frame['fields'];a=frame['amp0']
        if 'real' in opts:line(position,x,f['psi'].real/a,'Re[Ψ] / A₀',CYAN)
        if 'imag' in opts:line(position,x,f['psi'].imag/a,'Im[Ψ] / A₀',BLUE,dash='dot')
        if 'density' in opts:line(position,x,f['density']/a**2,'|Ψ|² / A₀²',PINK,fill=True)
        if 'components' in opts:
            line(position,x,f['incident'].real/a,'입사 부분 Re[Cg] / A₀',GREEN,width=1.5)
            line(position,x,f['reflected'].real/a,'반사 부분 Re[−Cg(−x)] / A₀',YELLOW,width=1.5,dash='dash')
        if 'phase' in opts:
            position.add_trace(go.Scatter(x=x,y=wall_phase_values(frame),mode='markers',name='|Ψ| / A₀ · 색 = 위상',
                marker=dict(size=5,color=np.angle(f['psi']),cmin=-np.pi,cmax=np.pi,
                    colorscale=[[0,'#ff5555'],[1/6,'#ffe35a'],[2/6,'#61e580'],[.5,'#48d7ff'],[4/6,'#7676ff'],[5/6,'#f071d5'],[1,'#ff5555']],
                    colorbar=dict(title='위상 arg Ψ',tickvals=[-np.pi,-np.pi/2,0,np.pi/2,np.pi],ticktext=['−π','−π/2','0','π/2','π'],thickness=12)),
                customdata=np.angle(f['psi']),hovertemplate='x=%{x:.2f}<br>|Ψ|/A₀=%{y:.3f}<br>위상=%{customdata:.3f} rad<extra></extra>'))
        position.add_vrect(x0=0,x1=3,fillcolor='#64748b',opacity=.18,line_width=0)
        position.add_vline(x=0,line_color='white',line_width=2)
        position.add_vline(x=frame['mean'],line_color='white',line_dash='dash',line_width=1.5)
        position.add_annotation(**wall_center_annotation(frame['mean']))
        position.update_xaxes(range=[float(x[0]),3])
        position.update_yaxes(range=[-1.15,1.25])
        line(spectral,frame['k'],frame['prob'],'현재 파수 분포 |A(k,t)|²',YELLOW,fill=True)
        spectral.add_vline(x=0,line_dash='dash',line_color='white')
        spectral.update_xaxes(range=[float(frame['k'][0]),float(frame['k'][-1])])
        spectral.update_yaxes(range=[0,2.2/(np.sqrt(2*np.pi)*sigma_k)])
        metrics=wall_metrics(frame)
        title='무한 퍼텐셜 벽에서의 반사 · x=0에서 Ψ=0'
        if k0<=2*sigma_k:
            warning='k₀가 σₖ에 비해 작아 초기 상태부터 양·음 파수 성분이 함께 중요합니다. 단일 중심의 벽 도달 시각은 반사 전체를 대표하지 않습니다. 반사 전후를 단순하게 보려면 k₀≥3σₖ 정도로 설정하세요.'
        note='x축은 −60~3으로 고정합니다. 화면 밖의 확률도 포함하여 계산하며 표시 범위에서 재정규화하지 않습니다.'
        knote='현재 Ψ(x,t)를 x≤0에서 사용하고 벽 안쪽을 0으로 연장한 Fourier 파수 분포입니다. 초기에는 +k 부근, 반사 후에는 −k 부근에 모입니다. 반사 중에는 두 방향이 함께 나타나며 간섭합니다. 그래프 밖의 작은 파수 꼬리는 생략하고 재정규화하지 않습니다. 운동량 측정과의 엄밀한 해석은 벽을 제거한 상태의 Fourier 분포로 이해할 수 있습니다.'
        concepts=[math_block('V(x)=\\begin{cases}0 & x<0\\\\ \\infty & x\\geq0\\end{cases}','\\Psi(0,t)=0'),
                  math_block('\\Psi(x,t)=C\\left[g(x,t)-g(-x,t)\\right]\\quad (x\\leq0)','\\Psi(x,t)=0\\quad(x>0)'),
                  html.P('g는 초기 중심 x₀=−18인 자유입자 가우스 해석해이고 C는 반직선에서 전체 확률을 1로 만드는 정확한 정규화 상수입니다. 두 항은 벽에서 항상 상쇄됩니다.'),
                  html.P(f'중심 파수의 자유 진행 속도는 k₀/m={k0/mass:.3f} v₀입니다. 벽과 상호작용하는 동안 실제 평균 위치와 평균 파수는 간섭을 반영해 변합니다.'),
                  html.P('벽 반사 모형의 σₖ와 Δx₀=1/(2σₖ)는 초기 가우스 g의 기준값입니다. 실제 벽 상태 C[g(x)−g(−x)]의 위치·파수 폭은 경계의 영향을 포함하므로 화면의 계산값으로 확인합니다.'),
                  html.P('모든 파수의 반사 확률은 1입니다. 자유입자의 분산관계 ω=ℏk²/(2m) 때문에 파동묶음이 퍼집니다. 무한 벽에서는 확률이 투과하거나 흡수되지 않습니다.'),
                  html.P(f'자유 가우스 기준의 퍼짐 시간 τ=2mΔx₀²={2*mass*sx0**2:.3f} T₀, 벽이 없을 때 g의 중심이 x=0에 도달할 시간 t_c=18m/k₀={18*mass/k0:.3f} T₀입니다. 자유 가우스 기준의 초기 속도 폭 Δv/v_g=σₖ/k₀={sigma_k/k0:.1%}입니다. τ가 t_c보다 짧으면 벽에 도착하기 전에 크게 퍼집니다.'),
                  html.P('반사 후 확률밀도의 잔물결은 아직 겹치는 입사 꼬리와 반사파의 간섭입니다. Re[Ψ]의 빠른 진동과 |Ψ|²의 잔물결을 구분해 보세요. 확률밀도만 켜거나 입사·반사 부분을 함께 켜면 원인을 확인할 수 있습니다.'),
                  html.P([html.A('해석해 근거: Belloni 등의 Gaussian wall reflection',href='https://arxiv.org/abs/quant-ph/0408182',target='_blank')],className='mini')]
        note+=' 반사 후 실제 평균 위치 ⟨x⟩가 −40에 도달할 때까지 재생합니다.'
        note+=' 색 점의 높이는 |Ψ|/A₀, 색은 전체 파동함수의 위상 arg Ψ입니다. −π와 π는 같은 색입니다. 진폭이 초기 기준의 1% 이하인 꼬리와 Ψ=0인 벽에서는 위상 색을 숨깁니다.'
        concepts.insert(0,html.P('색 자체는 위상 기준에 따라 바뀝니다. 전체에 같은 위상을 더해도 관측 결과는 같습니다. 물리적으로 중요한 것은 상대 위상(간섭)과 공간적 위상 변화(확률 흐름)입니다. 색이 시간에 따라 바뀌는 것만으로 입자의 이동을 판단하지 않습니다.'))
        concepts.insert(0,html.P('위상 φ(x,t)=atan2(Im Ψ, Re Ψ)=arg Ψ(x,t). 평면파는 φ=kx−ℏk²t/(2m)+φ₀이며, 반사파는 −kx−ℏk²t/(2m)+φ₀+π입니다. ℏ=1 단위를 사용합니다. 겹침 구간에서는 전체 파동함수의 위상을 표시합니다.'))
        data=dict(x=x,real=f['psi'].real,imag=f['psi'].imag,density=f['density'])
    elif mode=='mirror':
        x=MIRROR_X
        fields=mirror_fields(x,t,k0,pulse_sigma)
        if 'components' in opts:
            line(position,x,fields['incident'],'입사 Eᵢ',CYAN,width=1.7)
            line(position,x,fields['reflected'],'반사 Eᵣ',PINK,width=1.7)
        if 'real' in opts:line(position,x,fields['electric'],'합성 전기장 E',YELLOW,width=2.5)
        if 'imag' in opts:line(position,x,fields['magnetic'],'자기장 ZH',BLUE,dash='dot')
        if 'density' in opts:line(position,x,fields['energy']/2,'에너지 밀도 u / 2',GREEN,width=1.6)
        if 'envelope' in opts:
            ai,ar=mirror_envelopes(x,t,pulse_sigma)
            line(position,x,ai,'입사 포락선',CYAN,dash='dash',width=1)
            line(position,x,ar,'반사 포락선',PINK,dash='dash',width=1)
        for i,k_component in enumerate(reflection_wavenumbers(k0,pulse_sigma)):
            if f'k{i+1}' not in opts:continue
            p=mirror_overlay_component(t,k_component,k0,pulse_sigma)
            color=COMPONENT_COLORS[i]
            for key,label,dash in [('incident','입사',None),('reflected','반사','dash'),('electric','합성','dot')]:
                position.add_trace(go.Scatter(x=MIRROR_COMPONENT_X,y=p[key],mode='lines',
                    name=f'성분 {i+1} · k={k_component:.2f}',legendgroup=f'k{i+1}',showlegend=key=='incident',
                    line=dict(color=color,width=1.3 if key!='electric' else 1.0,dash=dash),opacity=.55 if key!='electric' else .35,
                    hovertemplate=f'성분 {i+1} · {label}<br>k={k_component:.3f}<br>E=%{{y:.4f}}<extra></extra>'))
        position.add_vrect(x0=0,x1=5,fillcolor='#64748b',opacity=.18,line_width=0,
                           annotation_text='이상적 반사체',annotation_position='top right')
        if 'directions' in opts:
            for annotation in mirror_direction_labels(k0,pulse_sigma,t):position.add_annotation(**annotation)
        position.add_vline(x=0,line_color='white',line_width=2)
        position.update_xaxes(range=[-30,5]);position.update_yaxes(range=[-2.1,2.1],title='E / E₀, ZH / E₀, u / 2')
        sk=1/(2*pulse_sigma)
        k=np.linspace(-8,8,1001)
        line(spectral,k,k_density(k,k0,sk),'입사 진행 성분의 스펙트럼',CYAN,fill=True)
        line(spectral,k,k_density(k,-k0,sk),'반사 진행 성분의 스펙트럼',PINK,fill=True)
        selected=reflection_wavenumbers(k0,pulse_sigma)
        heights=k_density(selected,k0,sk)
        component_circles(spectral,selected,heights,COMPONENT_COLORS,'대표 입사 성분 ●')
        component_circles(spectral,-selected,heights,COMPONENT_COLORS,'같은 성분의 반사 ○',reflected=True)
        spectral.update_yaxes(title='정규화된 진행 성분 스펙트럼')
        spectral.update_xaxes(range=[-8,8])
        metrics=[metric('에너지 반사율 R','1.000'),metric('투과율 T / 흡수율','0 / 0'),metric('전기장 반사계수 r','−1 · 위상 π'),metric('경계면 E(0,t)',f"{mirror_fields([0],t,k0,pulse_sigma)['electric'][0]:.3f}")]
        note='굵은 선은 전체 전기장 펄스입니다. 색 글씨는 펄스 중심의 진행 방향을 설명하는 표식으로, 경계 도달 전 +k →, 이후 ← −k를 나타냅니다. 무분산 진공에서는 모든 파수의 펄스가 같은 속도 c로 이동합니다. 경계와 겹치는 동안에는 입사·반사 펄스가 동시에 존재하며 표식의 전환 시각은 중심 도달 시각입니다. 필요하면 개별 평면파 성분을 켤 수 있습니다.'
        knote='원의 색은 위 전기장 펄스 패널의 대표 파수 성분과 같습니다. +k의 채운 원과 −k의 빈 원은 같은 성분의 입사·반사 쌍입니다. 시간에 따른 반사 완료율이 아닙니다. 실수 전기장의 Fourier 분포는 ±k의 대칭 성분을 함께 갖습니다.'
        concepts=[html.P('진공 중 비분산 펄스가 x=0의 이상적 완전 전기 반사 경계에 수직 입사합니다. 유전율만 다른 일반 유전체의 모형과 구분합니다.'),
                  math_block('E_i=f(x-t),\\qquad E_r=-f(-x-t)','E(0,t)=0'),
                  html.P('ω=c|k|, c=1. 반사된 파수는 −k, 주파수는 그대로이며 R(k)=1입니다. 파수를 바꾸어도 반사되는 에너지 비율은 변하지 않습니다.'),
                  html.P('양자 모형과 비교: 이 빛의 펄스는 퍼지지 않습니다. |Ψ|²의 위치 확률 해석은 이 전기장 그래프에 적용하지 않습니다.')]
        data=dict(x=x,**fields)
        title='전기장 펄스 · 진공 → 이상적 완전반사 경계'
    else:
        if mode=='plane':
            x=np.linspace(-15,15,1801);psi=np.exp(1j*(k0*x-k0*k0*t/(2*mass)))
            if 'real' in opts:line(position,x,psi.real,'Re[Ψ] / A',CYAN)
            if 'imag' in opts:line(position,x,psi.imag,'Im[Ψ] / A',BLUE,dash='dot')
            if 'density' in opts:line(position,x,np.ones_like(x),'|Ψ|² / |A|²',PINK)
            if 'phase' in opts:
                position.add_trace(go.Scatter(x=x,y=psi.real,mode='markers',
                    name='Re[Ψ] / A · 색 = 위상',
                    marker=dict(size=4,color=np.angle(psi),colorscale=PHASE_COLORS,
                        cmin=-np.pi,cmax=np.pi,colorbar=dict(title='위상',
                            tickvals=[-np.pi,-np.pi/2,0,np.pi/2,np.pi],
                            ticktext=['−π','−π/2','0','π/2','π'],thickness=14)),
                    customdata=np.angle(psi),
                    hovertemplate='x/L₀=%{x:.3f}<br>Re[Ψ]/A=%{y:.3f}<br>위상=%{customdata:.3f} rad<extra></extra>'))

            position.update_yaxes(range=[-1.15,1.25]);position.update_xaxes(range=[-15,15])
            position.add_annotation(x=.98,y=1.80,xref='paper',yref='y',
                                    text=f'위상 진행 →<br>v<sub>p</sub> = {k0/(2*mass):.3f} v<sub>0</sub>',
                                    showarrow=False,xanchor='right',yanchor='top',align='right',
                                    font=dict(size=18,color=YELLOW),bgcolor='#0d1b2e',borderpad=6)

            spectral.add_trace(go.Scatter(x=[k0,k0],y=[0,1],mode='lines+markers',name='단일 파수',line=dict(color=YELLOW,width=3)))
            if mode=='build':component_circles(spectral,[k0],[1],[COMPONENT_COLORS[0]],'대표 성분 ●')
            spectral.update_xaxes(range=[-2,8]);spectral.update_yaxes(title='단일 파수 · δ 분포의 개념 표시',showticklabels=False)
            metrics=[metric('파장 λ / L₀',f'{2*np.pi/k0:.3f}'),metric('운동량 p / (ℏ/L₀)',f'{k0:.3f}'),metric('파수 불확정성 Δk','0 · 이상적 극한'),metric('위치 불확정성 Δx','비국소화')]
            concepts=[math_block('\\omega=\\frac{\\hbar k^2}{2m},\\qquad p=\\hbar k','v_p=\\frac{\\omega}{k}=\\frac{\\hbar k}{2m}'),html.P('위상은 이동하지만 |Ψ|²는 일정합니다. 이상적 평면파는 무한한 공간에서 정규화할 수 없으며, 표시 구간의 폭을 Δx로 해석하지 않습니다.')]
            concepts.append(math_block('\\phi(x,t)=\\arg\\Psi(x,t)=kx-\\omega t\\quad(\\mathrm{mod}\\ 2\\pi)'))
            concepts.append(dcc.Markdown('이 그림에서는 $A$를 양의 실수로 잡았습니다. **색은 복소 파동함수의 위상**, 파형의 높이는 $\\operatorname{Re}\\Psi/A$입니다. 실수부의 모든 마루는 위상 $0$ (또는 $2\\pi$의 정수배)이므로 같은 색입니다. 골은 $\\pi$와 $-\\pi$로 같은 위상이며 같은 색입니다. 높이가 0인 지점도 $|\\Psi|=A$이므로 위상은 정의됩니다. 색은 확률이나 파수 성분의 구분을 뜻하지 않습니다.',mathjax=True))
            concepts.append(dcc.Markdown('위상속도 $v_p=\\frac{\\omega}{k}=\\frac{\\hbar k}{2m}$는 같은 위상(마루 등)이 이동하는 속도입니다. 이 평면파가 국소화된 입자의 위치를 나타내는 것은 아닙니다. 표시값은 속도 단위 $v_0=\\frac{\\hbar}{M_0L_0}$에 대한 비율이며, k와 m은 각각 L₀⁻¹, M₀를 단위로 한 입력값입니다. 같은 k에서 질량이 클수록 위상속도는 느려집니다.',mathjax=True))
            position.update_xaxes(title='위치 x / L₀')
            spectral.update_xaxes(title='파수 k · 단위 1/L₀')
            concepts.append(dcc.Markdown(r'$L_0$는 위치를 나타내기 위해 선택한 **기준 길이**, $M_0$는 질량을 나타내기 위해 선택한 **기준 질량**입니다. 특정 물체의 크기나 질량으로 고정되어 있지 않습니다. 화면의 $x=1$은 실제 길이 $L_0$, $m=1$은 실제 질량 $M_0$, 파수 입력 $k=1$은 실제 파수 $1/L_0$에 해당합니다.',mathjax=True))
            concepts.append(math_block(r'\frac{\lambda}{L_0}=\frac{2\pi}{k}',r'T_0=\frac{M_0L_0^2}{\hbar},\qquad v_0=\frac{L_0}{T_0}=\frac{\hbar}{M_0L_0}'))
            concepts.append(html.P('파장 지표는 L₀ 단위의 값입니다. 예를 들어 파장 값이 2.094이면 실제 파장은 2.094 L₀입니다. 여기의 k는 입력한 무차원 파수이며, 실제 파수를 쓰면 λ=2π/k입니다.'))
            note='위치 그래프는 임의 진폭 A 기준입니다. 평면파의 상수 |Ψ|²는 정규화된 단일 입자의 확률밀도가 아닙니다.'
            knote='한 파수만 포함한다는 개념을 막대로 표시합니다. 막대의 높이는 유한한 확률밀도 값이 아닙니다.'
            data=dict(x=x,real=psi.real,imag=psi.imag,density=np.abs(psi)**2);title='평면파 · 위상과 밀도'
        else:
            width=gaussian_width(t,sx0,mass) if mode=='free' else sx0
            mean=k0*t/mass if mode=='free' else 0.0
            span=max(15,6*width)
            if mode=='free' and 'follow' not in opts:
                # Fix the camera to a domain covering the full selected experiment.
                lo,hi,_=free_viewport(k0,mass,sigma_k)
                x=np.linspace(lo,hi,2601)
            else:x=np.linspace(mean-span,mean+span,1801)
            psi=gaussian_packet(x,t if mode=='free' else 0,k0,sx0,mass)
            ref=gaussian_packet(x,0,k0,sx0,mass)
            amp0=(2*np.pi*sx0*sx0)**(-.25)
            rho0=amp0**2
            if mode=='build':
                shown,ks,amps=progressive_superposition(x,int(count),k0,sigma_k)
                dk=8*sigma_k/100
                halfspan=(count//2)*dk
                probs=amps**2/np.sum(amps**2)
                sample_sd=float(np.sqrt(np.sum(probs*(ks-k0)**2)))
                warning=f'N={count}: k={ks[0]:.2f}~{ks[-1]:.2f}, 성분 간격 Δk_step={dk:.3f}. N을 늘리면 중심에서 바깥쪽 파수 성분이 추가됩니다.'
                if 'reference' in opts:
                    line(position,x,np.abs(psi)**2/rho0,'모든 k · 가우스 기준 |Ψ|²/A₀²',GREEN,dash='dash',width=1.6)
            else:
                shown=psi
                if 'reference' in opts:line(position,x,np.abs(ref)**2/rho0,'t=0 확률밀도 기준',GREEN,dash='dash',width=1.4)
            finite_label=f'N={count}개 · ' if mode=='build' else ''
            if 'real' in opts:line(position,x,shown.real/amp0,finite_label+'Re[Ψ] / A₀',CYAN)
            if 'imag' in opts:line(position,x,shown.imag/amp0,finite_label+'Im[Ψ] / A₀',BLUE,dash='dot')
            if 'density' in opts:line(position,x,np.abs(shown)**2/rho0,finite_label+'|Ψ|² / A₀²',PINK,fill=True)
            if 'envelope' in opts:
                line(position,x,np.abs(psi)/amp0,'모든 k · 가우스 포락선 |Ψ|/A₀',YELLOW,dash='dot',width=1)
                line(position,x,-np.abs(psi)/amp0,'모든 k · 음의 가우스 포락선',YELLOW,dash='dot',width=1)
            position.add_vline(x=mean,line_color='white',line_dash='dash')
            label_left=mean>float(x[0])+.82*(float(x[-1])-float(x[0]))
            position.add_annotation(x=mean,xref='x',y=.98,yref='paper',
                                    text=f'⟨x⟩<br>v<sub>g</sub> = {k0/mass:.3f} v<sub>0</sub>' if mode=='free' else '가우스 기준 중심 x=0',
                                    showarrow=False,xanchor='right' if label_left else 'left',
                                    xshift=-8 if label_left else 8,yanchor='top',align='left',
                                    font=dict(size=17,color='white'),bgcolor='#0d1b2e',borderpad=4)
            position.add_vrect(x0=mean-width,x1=mean+width,fillcolor=CYAN,opacity=.055,line_width=0)
            position.update_xaxes(range=[float(x[0]),float(x[-1])])
            ymax=max(1.2,float(np.max(np.abs(shown)**2/rho0))*1.1)
            position.update_yaxes(range=[-1.15,ymax])
            k=np.linspace(k0-5*sigma_k,k0+5*sigma_k,601)
            line(spectral,k,k_density(k,k0,sigma_k),'모든 k · 가우스 파수 분포',YELLOW,fill=True)
            if mode=='build':
                stems_x=np.column_stack((ks,ks,np.full_like(ks,np.nan))).ravel()
                stems_y=np.column_stack((np.zeros_like(ks),amps**2,np.full_like(ks,np.nan))).ravel()
                spectral.add_trace(go.Scatter(x=stems_x,y=stems_y,mode='lines',name=f'포함된 {count}개 파수',line=dict(color=CYAN,width=2)))
                spectral.add_trace(go.Scatter(x=ks,y=amps**2,mode='markers',name='가우스 파수 분포 표본 |A(kⱼ)|²',marker=dict(color=CYAN,size=6),showlegend=False))
                chosen=representative_indices(len(ks))
                component_circles(spectral,ks[chosen],amps[chosen]**2,COMPONENT_COLORS[:len(chosen)],'대표 성분 ●')
                spectral.add_vrect(x0=ks[0],x1=ks[-1],fillcolor=CYAN,opacity=.08,line_width=0)
            spectral.add_vline(x=k0,line_dash='dash',line_color='white')
            spectral.update_xaxes(range=[k0-5*sigma_k,k0+5*sigma_k])
            prefix='연속 목표 ' if mode=='build' else ''
            metrics=[metric(prefix+'평균 위치 ⟨x⟩',f'{mean:.3f}'),metric(prefix+'위치 표준편차 Δx',f'{width:.3f}'),metric(prefix+'파수 표준편차 Δk',f'{sigma_k:.3f}'),metric('현재 Δx(t)·Δp / ℏ' if mode=='free' else prefix+'Δx·Δp / ℏ',f'{width*sigma_k:.5f}')]
            if mode=='build':
                metrics=[metric('포함된 평면파 N',str(count)),metric('파수 범위 반폭',f'{halfspan:.3f}'),metric('포함 성분의 파수 표준편차',f'{sample_sd:.3f}'),metric('가우스 기준의 초기 Δx',f'{sx0:.3f}')]
                concepts=[math_block('\\Psi(x)=\\frac{1}{\\sqrt{2\\pi}}\\int_{-\\infty}^{\\infty}A(k)e^{ikx}\\,dk'),math_block('\\Delta x=\\frac{1}{2\\sigma_k},\\quad \\Delta k=\\sigma_k,\\quad\\Delta x\\,\\Delta p=\\frac{\\hbar}{2}'),
                          dcc.Markdown(r'N을 늘리면 일정한 성분 간격 $\Delta k_{\mathrm{step}}=\frac{8\sigma_k}{100}$을 유지하면서 중심에서 양쪽으로 평면파를 추가합니다. 더 넓은 파수 범위가 간섭에 참여하면서 중심의 파군이 형성됩니다. N=101에서는 중심 파수 양쪽으로 파수 폭의 4배까지, 즉 k₀−4σₖ부터 k₀+4σₖ까지 101개 성분을 더합니다. 그 바깥의 성분은 가중치가 작아 생략하므로, 모든 파수를 포함한 정확한 가우스 파동묶음과는 작은 차이가 있는 이산 근사입니다.',mathjax=True),
                          html.P('N이 충분히 커지면 작은 진폭의 바깥 성분만 추가되므로 변화가 작아집니다. 이때 σₖ를 바꾸면 국소화 정도의 차이를 계속 관찰할 수 있습니다.'),
                          html.P('유한한 개수의 평면파 합은 주기적인 반복 무늬를 만들 수 있습니다. 화면에서 잘라낸 분포를 정규화하여 불확정성 곱을 계산하지 않습니다.')]
                note=f'실선은 N={count}개 평면파의 합입니다. 파란색은 실수부, 분홍색은 합성 파동함수의 절댓값 제곱이며 실수부만 제곱한 값이 아닙니다. 초록 점선은 모든 k를 포함한 가우스 파동묶음의 비교 기준입니다. A₀=|Ψ(x₀,0)|는 이 가우스 기준의 초기 중심 진폭 크기입니다.'+' 확률밀도는 그 제곱 A₀²으로 나눕니다. 선의 상대 높이는 서로 다른 물리량의 단위를 같게 만들지 않습니다.'
                knote='노란 곡선은 가우스 파수 확률밀도 |A(k)|², 파란 막대는 현재 포함한 N개 파수에서의 표본 |A(kⱼ)|²입니다. 막대 높이는 합성파의 계수 |cⱼ|² 자체가 아니며 실제 합성 계수는 cⱼ=A(kⱼ)Δk_step/√(2π)입니다. 대표 성분의 막대 꼭대기 원과 아래 개별 파형은 같은 색입니다. N을 늘리면 일정한 간격으로 양쪽 성분이 추가됩니다. σₖ는 목표 분포의 표준편차입니다.'
                title='평면파를 더해 파동묶음 만들기'
            else:
                vg=k0/mass;vp=k0/(2*mass)
                concepts=[math_block('\\langle x\\rangle=\\frac{k_0t}{m}','\\Delta x(t)=\\Delta x_0\\sqrt{1+\\left(\\frac{t}{2m\\Delta x_0^2}\\right)^2}\\quad(\\hbar=1)'),html.P(f'파동묶음의 중심 이동 속도는 v_g={vg:.3f} v₀, 중심 파수의 위상속도는 v_p={vp:.3f} v₀입니다. 자유입자의 분산관계 ω=k²/(2m)는 비선형이므로, 각 파수 성분의 상대 위상이 시간에 따라 변하면서 파동묶음이 퍼집니다.'),
                          html.P('단위: 질량과 파수의 기준 단위는 각각 M₀와 1/L₀입니다. x축 값 1은 기준 길이 L₀, 시간 값 1은 기준 시간 T₀에 해당합니다. T₀=M₀L₀²/ℏ, 속도 단위는 v₀=L₀/T₀=ℏ/(M₀L₀)입니다. 실제 기준 길이·질량을 선택하면 SI 단위로 환산할 수 있습니다.'),
                          html.P('파수 확률분포는 시간에 따라 바뀌지 않습니다. Δx·Δp는 t=0에서 ℏ/2이고 이후 커집니다. 확률 총합은 항상 1입니다.'),
                          html.P('파동묶음의 퍼짐은 확률분포의 변화입니다. 실제 입자 하나가 공간에서 쪼개진다는 뜻으로 해석하지 않습니다.')]
                concepts.append(math_block('\\frac{\\Delta x(t)\\,\\Delta p}{\\hbar}=\\frac12\\sqrt{1+\\left(\\frac{t}{\\tau}\\right)^2}', rf'\tau=2m\Delta x_0^2={2*mass*sx0**2:.3f}\,T_0\quad(\hbar=1)'))
                v_nm,t_nm=electron_units(1e-9)
                concepts.append(html.Details([
                    html.Summary('단위 설명 · 전자와 1 nm 환산 예시'),
                    html.P('기준 질량 M₀=mₑ≈9.109×10⁻³¹ kg으로 선택한 예입니다. 화면에서 m=1일 때 전자 질량에 해당합니다. L₀는 시료 크기가 아니라 위치 좌표의 기준 길이입니다.'),
                    html.Table([html.Thead(html.Tr([html.Th('기준 길이 L₀'),html.Th('속도 단위 v₀'),html.Th('시간 단위 T₀')])),
                                html.Tbody([html.Tr([html.Td('1 nm · 미시적 예시'),html.Td(f'{v_nm:.3e} m/s'),html.Td(f'{t_nm*1e15:.3f} fs')])])],className='unit-table'),
                    html.P('x축 값이 1 증가하면 실제 거리는 L₀만큼 증가합니다. 시간 값 t=1은 실제 시간 T₀에 해당하며, 화면 재생 시간과는 다릅니다.'),
                    html.P(f'L₀=1 nm로 해석하면 현재 v_g={vg:.3f} v₀≈{vg*v_nm:.3e} m/s입니다. 이는 단위 환산 예시이며 그래프의 계산 조건을 바꾸지 않습니다.'),
                    html.P('고체의 전자 파동을 미시적으로 설명하려면 원자·나노미터 규모를 고려하는 것이 자연스럽습니다. 예를 들어 실리콘의 격자상수는 약 0.543 nm입니다. 이 수업의 자유입자 예시는 L₀=1 nm로 해석할 수 있지만, 실제 고체의 전자를 정량적으로 다루려면 물질과 밴드에 따른 유효질량·분산관계를 고려해야 합니다.'),
                    html.P([html.A('수치 근거: NIST/CODATA',href='https://physics.nist.gov/cuu/pdf/wall_2022.pdf',target='_blank'),' · ',html.A('격자상수·유효질량: TU Graz 실리콘 자료',href='https://lampz.tugraz.at/~hadley/memm/materials/silicon/silicon.php',target='_blank')],className='mini')]))
                note='흰 점선: 평균 위치 ⟨x⟩. 밝은 음영: 평균 위치 ± 위치 표준편차 Δx(t). 가우스 분포의 위치 확률 약 68%가 이 구간에 들어가며, 음영은 경계가 아니므로 밖에도 위치 확률이 있습니다. 정확한 연속 가우스 해석해를 표시합니다. t=0 기준 높이를 고정해 퍼질 때 봉우리 높이가 낮아지는 것을 보여줍니다. 고정 좌표에서는 중심이 오른쪽 끝까지 이동하도록 재생합니다. 화면 밖의 확률분포도 존재하며 표시 구간에서 다시 정규화하지 않습니다.'
                knote='연속적인 가우스 파수 확률분포의 모든 k(−∞<k<∞)를 사용합니다. N개의 유한합 대신 전체 분포의 정확한 해석해를 계산합니다. 그래프는 k₀±5σₖ 구간만 표시합니다. |A(k)|²는 시간에 무관합니다.'
                concepts.insert(0,html.P('연속 가우스 모형 · 모든 k∈(−∞,∞) 사용 · N 제한 없음',className='formula'))
                title='자유입자 · 복소 파동함수와 위치 확률밀도'
            data=dict(x=x,real=shown.real,imag=shown.imag,density=np.abs(shown)**2)
    if mode in ('build','free','wall','beat'):
        concepts.append(dcc.Markdown(r'**파수 폭을 읽는 법:** 중심 파수 $k_0$는 분포의 중심, 파수 폭 $\sigma_k$는 그 주변에 파수들이 얼마나 넓게 퍼져 있는지를 나타냅니다. 폭이 작으면 비슷한 파장의 파들이, 폭이 크면 더 다양한 파장의 파들이 중첩됩니다. 가우스 기준에서는 $\Delta x_0=\frac{1}{2\sigma_k}$이므로 파수 폭이 클수록 **초기** 위치 분포는 좁아집니다.',mathjax=True))
        concepts.append(dcc.Markdown(r'정확히는 $\sigma_k$가 기준 가우스 파수 확률밀도 $|A(k)|^2$의 **표준편차**입니다. 이는 분포의 퍼진 정도를 나타내며 포함되는 파수 범위의 양 끝을 정하는 값은 아닙니다. 중첩 실험에서는 연속 목표 분포의 폭, 벽 반사에서는 초기 가우스 $g$의 폭이며 실제로 표시한 상태의 폭과 구분합니다.',mathjax=True))
    if mode=='wall':
        concepts.insert(0,html.Div([math_block('A_0\\equiv|\\Psi(-18,0)|'),html.P('실제 벽 상태의 초기 기준 진폭 크기입니다. 확률밀도는 고정된 초기 진폭의 제곱으로 나누어 표시합니다.')]))
    if mode in ('build','free'):
        concepts.insert(0,math_block('A_0\\equiv|\\Psi(x_0,0)|=\\left(2\\pi\\Delta x_0^2\\right)^{-1/4}'))
        if mode=='free':note+=' A₀는 t=0 중심 진폭 크기입니다. |Ψ|²/A₀²의 초기 최대값은 1이며, 퍼짐에 따라 감소합니다.'
    if mode=='build':
        concepts[0:0]=[
            html.P('그래프 읽기 · 현재 t=0에서 각 평면파의 진폭을 aⱼ라 하면:'),
            math_block('\\operatorname{Re}\\Psi=\\sum_j a_j\\cos(k_jx)'),
            math_block('\\operatorname{Im}\\Psi=\\sum_j a_j\\sin(k_jx)'),
            math_block('|\\Psi|^2=(\\operatorname{Re}\\Psi)^2+(\\operatorname{Im}\\Psi)^2'),
            html.P('각 합을 제곱할 때 서로 다른 파수 사이의 간섭항이 생깁니다. 개별 성분의 확률밀도를 따로 더한 값과는 다릅니다.')]
        note+=' Re[Ψ]는 코사인의 가중합, Im[Ψ]는 사인의 가중합입니다. |Ψ|²는 두 합을 각각 제곱하여 더한 값입니다.'
    fit_vertical_axis(position,axis_mode,zoom)
    if mode=='plane':
        low,high=position.layout.yaxis.range
        position.update_yaxes(range=[low,max(1.95,high)])
    fit_vertical_axis(spectral,'auto',1.0)
    if axis_mode=='auto':
        note+=' 현재 표시된 곡선에 맞춰 y축을 자동 조정합니다. 시간별 높이는 y축 눈금으로 비교하고, 봉우리 감소를 같은 축에서 보려면 초기 기준을 선택하세요.'
    if zoom!=1:
        note+=f' y축 {zoom:g}배 확대 중입니다. 축 바깥의 부분은 잘릴 수 있습니다.'
    return position,spectral,metrics,concepts,question_items(mode,count),warning,note,knote,title,data


PHASE_COLORS=[[0,'#ff5555'],[1/6,'#ffe35a'],[2/6,'#61e580'],[.5,'#48d7ff'],[4/6,'#7676ff'],[5/6,'#f071d5'],[1,'#ff5555']]


def phase_clock_arrow(psi,a):
    """Unit phase arrow; its length is a display convention, not amplitude."""
    phase=float(np.angle(psi))
    points=np.array([[0,.025],[.52,.025],[.52,.11],[.85,0],[.52,-.11],[.52,-.025],[0,-.025],[0,.025]])
    rotation=np.array([[np.cos(phase),-np.sin(phase)],[np.sin(phase),np.cos(phase)]])
    points=points@rotation.T
    if abs(psi)/a<=.01:points[:]=np.nan
    return points[:,0],points[:,1]


def wall_phase_detail(k0,sk,mass,t):
    x=np.linspace(-8,0,241)
    fields=wall_packet(x,t,k0,1/(2*sk),mass)
    a=wall_frame(k0,sk,mass,t)['amp0']
    fig=make_subplots(rows=3,cols=2,shared_xaxes=True,vertical_spacing=.13,horizontal_spacing=.09,
                      column_widths=[.72,.28],specs=[[{},{}],[{},{}],[{},None]],
                      subplot_titles=['입사 부분 →','벽에서의 입사 위상','반사 부분 ← · 반사계수 −1','벽에서의 반사 위상','합성 파동함수 · 벽에서 상쇄'])
    for row,key in enumerate(['incident','reflected','psi'],1):
        psi=fields[key];amplitude=np.abs(psi)/a
        height=np.where(amplitude>.01,amplitude,np.nan)
        fig.add_trace(go.Scatter(x=x,y=height,mode='markers',showlegend=False,
            marker=dict(size=6,color=np.angle(psi),colorscale=PHASE_COLORS,cmin=-np.pi,cmax=np.pi,showscale=False),
            customdata=np.angle(psi),hovertemplate='x=%{x:.2f}<br>|Ψ|/A₀=%{y:.3f}<br>위상=%{customdata:.3f} rad<extra></extra>'),row=row,col=1)
        fig.add_vline(x=0,line_color='white',row=row,col=1)
        fig.update_yaxes(range=[-.04,1.25],title_text='|Ψ| / A₀',row=row,col=1)
        fig.update_xaxes(range=[-8,.15],gridcolor='#1c3550',row=row,col=1)
    theta=np.linspace(-np.pi,np.pi,181)
    for row in [1,2]:
        fig.add_trace(go.Scatter(x=np.cos(theta),y=np.sin(theta),mode='markers',showlegend=False,
            marker=dict(size=7,color=theta,colorscale=PHASE_COLORS,cmin=-np.pi,cmax=np.pi),
            customdata=theta,hovertemplate='위상=%{customdata:.3f} rad<extra></extra>'),row=row,col=2)
        axis='x2' if row==1 else 'x4'
        fig.update_xaxes(range=[-1.55,1.55],visible=False,matches=None,row=row,col=2)
        fig.update_yaxes(range=[-1.35,1.35],visible=False,scaleanchor=axis,scaleratio=1,constrain='domain',row=row,col=2)
        for px,py,label in [(1.25,0,'0'),(-1.25,0,'−π / π'),(0,1.2,'π/2'),(0,-1.2,'−π/2')]:
            fig.add_annotation(x=px,y=py,text=label,showarrow=False,font=dict(size=16,family='STIX Two Math, Cambria Math, Times New Roman, serif',color='#f4f8ff'),row=row,col=2)
    for row,key in enumerate(['incident','reflected'],1):
        ax,ay=phase_clock_arrow(fields[key][-1],a)
        phase=float(np.angle(fields[key][-1]))
        fig.add_trace(go.Scatter(x=ax,y=ay,mode='lines',fill='toself',fillcolor='white',line=dict(color='white',width=1),
            showlegend=False,text=[f'벽 x=0 · 위상={phase:.3f} rad']*len(ax),hovertemplate='%{text}<extra></extra>'),row=row,col=2)
    fig.update_xaxes(title_text='벽 근처 위치 x',row=3,col=1)
    fig.update_layout(template='plotly_dark',height=700,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#071524',
                      margin=dict(l=70,r=30,t=65,b=55),font=dict(color='#d2e1f0',size=17),uirevision='wall-phase-clocks')
    title='벽 근처 위상 확대 · 입사 / 반사 / 합성'
    note='원 둘레의 색은 위상 범례이며 오른쪽이 0, 반시계 방향이 양의 위상입니다. −π와 π는 같은 지점입니다. 흰 화살표는 같은 시각 벽 x=0의 입사·반사 위상을 나타내며 항상 π 차이로 반대 방향을 가리킵니다. 화살표 길이는 진폭을 뜻하지 않습니다. 벽에서 각 부분의 진폭이 초기 기준의 1% 이하이면 화살표를 숨깁니다. 합성파는 벽에서 0이므로 위상 화살표를 정의할 수 없습니다. 세 파형의 진폭 눈금과 위상 색 기준은 같습니다.'
    return fig,title,note,700


def component_figure(mode,k0,sigma_k,count,pulse_sigma,t):
    """Separate actual Fourier plane-wave terms, with no pulse envelopes."""
    if mode not in ('build','mirror'):
        return go.Figure(),'', '',0
    if mode=='build':
        x=np.linspace(-15,15,1201)
        _,ks,amps=progressive_superposition([0],count,k0,sigma_k)
        indices=representative_indices(len(ks))
        dk=8*sigma_k/100
        norm=(2*np.pi/(4*sigma_k*sigma_k))**(-.25)
        coeff=amps*dk/np.sqrt(2*np.pi)/norm
        titles=[f'k={ks[i]:.3f} · 기여 진폭={coeff[i]:.4f}' for i in indices]
        fig=make_subplots(rows=len(indices),cols=1,shared_xaxes=True,vertical_spacing=.07 if len(indices)>1 else 0,subplot_titles=titles)
        for row,j in enumerate(indices,1):
            y=coeff[j]*np.cos(ks[j]*x)
            fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name=f'대표 성분 {row}',line=dict(color=COMPONENT_COLORS[row-1],width=2.5),showlegend=False),row=row,col=1)
            fig.update_yaxes(range=[-max(coeff[j]*1.2,.001),max(coeff[j]*1.2,.001)],nticks=3,row=row,col=1)
        title=f'현재 N={count}에 포함된 개별 평면파 · 대표 {len(indices)}개'
        note='각 행은 위 합성파에 실제로 더해지는 평면파의 실수부입니다. 행마다 y축 범위가 다르므로 진폭은 눈금으로 비교하세요. 전체 N개 중 최대 5개를 표시합니다. 성분을 늘릴 때 파수 범위가 넓어지는 모습과 중심 부근의 위상 정렬을 비교하세요.'
    else:
        x=MIRROR_COMPONENT_X
        ks=reflection_wavenumbers(k0,pulse_sigma)
        parts=[mirror_component(x,t,float(k),pulse_sigma,k0) for k in ks]
        titles=[f'k={k:.3f} → −k={-k:.3f} · 상대 스펙트럼 진폭={p["amplitude"]:.3f}' for k,p in zip(ks,parts)]
        fig=make_subplots(rows=5,cols=1,shared_xaxes=True,vertical_spacing=.055,subplot_titles=titles)
        for row,(k,p) in enumerate(zip(ks,parts),1):
            color=COMPONENT_COLORS[row-1]
            for key,label,dash in [('incident','입사 → · 실선',None),('reflected','← 반사 · 파선','dash'),('electric','합성 · 점선','dot')]:
                fig.add_trace(go.Scatter(x=x,y=p[key],name=label,mode='lines',legendgroup=key,showlegend=row==1,
                                        line=dict(color=color,width=2.4 if key!='electric' else 1.5,dash=dash),opacity=.65 if key=='electric' else 1),row=row,col=1)
            fig.add_vline(x=0,line_color='white',row=row,col=1)
            fig.update_yaxes(range=[-2.2,2.2],nticks=3,row=row,col=1)
        title='파수 성분별 완전반사 · 입사파, 반사파, 합성파'
        note='각 행의 색은 파수 그래프 원의 색과 같습니다. 같은 성분 안에서는 실선=입사, 파선=반사, 점선=합성입니다. 대표 양의 파수 5개를 실제 Fourier 분해에서 선택합니다. 개별 성분은 공간 전체에 퍼진 평면파이고 반사 성분도 항상 존재합니다. 펄스 도착은 모든 성분의 중첩으로 나타납니다. 각 성분에서 ω는 그대로, k는 −k로 바뀌고 Eᵣ(0,t)=−Eᵢ(0,t)입니다.'
    height=220 if len(titles)==1 else 180*len(titles)+120
    fig.update_layout(template='plotly_dark',height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#071524',margin=dict(l=65,r=30,t=85,b=55),font=dict(size=17,color='#d2e1f0'),legend=dict(orientation='h',y=1.07,x=0),uirevision=f'{mode}-components')
    fig.update_xaxes(gridcolor='#1c3550',zerolinecolor='#496078',range=[float(x[0]),float(x[-1])])
    fig.update_yaxes(gridcolor='#1c3550',zerolinecolor='#496078')
    fig.update_xaxes(title_text='위치 x',row=len(titles),col=1)
    for annotation,color in zip(fig.layout.annotations,COMPONENT_COLORS):
        annotation.font.color=color
    return fig,title,note,height


app=Dash(__name__,assets_folder=str(Path(__file__).parent/'assets'))
app.title='양자역학 시뮬레이션 실험실'
app.layout=html.Div([
    html.Header([html.Div([html.Div('AI 기반 물리 시뮬레이션 교육 · 전남대학교 물리교육과',className='eyebrow'),html.H1('양자역학 시뮬레이션 실험실'),html.Div('Quantum Mechanics Lab',className='english-title'),html.P('퍼텐셜에 따른 에너지 고유상태와 파동함수의 중첩, 시간 변화를 직접 탐구합니다.')]),html.Div(id='unit-label',className='unit')]),
    dcc.Tabs(id='system',value='well',children=[dcc.Tab(label=label,value=value,className='lesson-tab',selected_className='lesson-tab-selected') for value,label in [('well','01 무한 퍼텐셜 우물'),('oscillator','02 조화진동자'),('waves','03 파동과 분산'),('free','04 자유입자')]],className='lesson-tabs'),
    html.Div([html.Span('하위 실험',className='mini'),dcc.Tabs(id='mode',value='well',children=[dcc.Tab(label='에너지 고유상태 · 중첩 · 시간 변화',value='well',className='lesson-tab',selected_className='lesson-tab-selected')],className='lesson-tabs experiment-tabs')],className='experiment-nav'),
    html.Div([html.H2(id='lesson-title'),html.P(id='lesson-desc')],className='intro'),
    html.Main([
        html.Aside([html.H3('실험 조건'),html.Div([html.Button('소수 성분',id='few'),html.Button('국소화된 파군',id='packet'),html.Button('더 좁은 파군',id='narrow')],id='presets',className='presets'),
            html.Div([
                slider('k0','중심 파수 k₀',.5,6,.1,3,'초기 중심 운동량 p₀=ℏk₀'),
                slider('sigma-k','파수 폭 σₖ',.15,2,.05,.7,'파수들이 중심 파수 주변에 퍼져 있는 정도입니다. 가우스 기준으로 폭이 클수록 다양한 파장이 포함되고 초기 파동묶음은 좁아집니다.'),
                slider('count','평면파 성분 개수 N',1,101,2,7,'중심 파수부터 양쪽으로 성분을 추가'),
                slider('mass','입자 질량 m',.5,4,.1,1,'같은 k₀·σₖ·시간에서 질량이 크면 이동과 퍼짐이 느립니다.'),
                slider('pulse-width','빛의 펄스 폭 σ',1,4,.1,2.5,'전기장 포락선 exp[−(x−x₀)²/(4σ²)]'),
            ],className='controlgrid'),
            ho.controls(),ft.controls(),well.controls(),rect.controls(),beat.controls(),
            slider('time','시간 t',0,10,.01,0),
            html.Div([html.Button('▶ 재생',id='play',className='primary'),html.Button('시간 초기화',id='restart')],className='buttons',id='playback-controls'),
            html.Div([html.Label('재생 속도',htmlFor='speed'),dcc.RadioItems(id='speed',options=[{'label':'0.5배','value':.5},{'label':'1배','value':1},{'label':'3배','value':3}],value=1,inline=True,className='speed-choice')],className='control',id='speed-control'),
            dcc.Checklist(id='options',value=['real','density','components','follow','reference'],className='checks',labelStyle={'display':'flex','gap':'8px'}),
            html.P(id='status',className='status',role='status')],className='controls'),
        html.Section([html.Div([html.Button('현재 데이터 CSV 저장',id='export'),html.Span('그래프 우상단 카메라 아이콘: PNG 저장',className='mini')],className='toolbar'),
            html.Div([html.Div([html.Label('위치 그래프 y축 보기'),dcc.RadioItems(id='axis-mode',options=[{'label':'현재 파형에 맞춤','value':'auto'},{'label':'초기 기준 유지','value':'fixed'}],value='fixed',inline=True,className='speed-choice')]),
                      html.Div([html.Label('y축 확대'),dcc.Slider(id='y-zoom',min=.5,max=4,step=.25,value=1,marks={.5:'0.5배',1:'1배',2:'2배',4:'4배'})])],className='axis-controls'),
            html.Div(id='vector-intro-slot',children=[]),
            html.Article([html.H3(id='position-title'),html.Div(id='graph-equations',style={'display':'none'}),dcc.Graph(id='position',responsive=True,className='position-graph',config={'displaylogo':False,'modeBarButtonsToRemove':['select2d','lasso2d'],'toImageButtonOptions':{'format':'png','filename':'wave-position','scale':2}}),html.P(id='position-note',className='plotnote')],className='plot',id='position-panel'),
            html.Article([html.H3(id='spectrum-title'),dcc.Graph(id='spectrum',responsive=True,className='spectrum-graph',config={'displaylogo':False,'modeBarButtonsToRemove':['select2d','lasso2d'],'toImageButtonOptions':{'format':'png','filename':'wave-spectrum','scale':2}}),html.P(id='spectrum-note',className='plotnote')],className='plot',id='spectrum-panel'),
            html.Article([html.H3(id='component-title'),dcc.Graph(id='component-plot',responsive=True,className='component-graph',config={'displaylogo':False,'modeBarButtonsToRemove':['select2d','lasso2d'],'toImageButtonOptions':{'format':'png','filename':'wave-components','scale':2}}),html.P(id='component-note',className='plotnote')],id='component-panel',className='plot',style={'display':'none'}),
            html.Section(id='metrics',className='metrics'),html.Section([html.H3('핵심 관계'),html.Div(id='concept'),html.H3('예측 · 관찰 · 설명'),html.Ol(id='questions',className='questions')],className='concept')],className='content',id='experiment-content')]),
    html.Details([html.Summary('모형과 그래프 읽는 법'),dcc.Markdown(r'수업 표기: 일반 파동함수는 $\Psi(x,t)$, 에너지 고유함수는 $\psi_n(x)$입니다. 따라서 $\Psi(x,t)=\sum_n c_n\psi_n(x)e^{-iE_nt/\hbar}$로 씁니다. $A_0$는 그래프의 초기 진폭 기준이며 고유함수 $\psi_0$와 구분합니다. 푸리에 변환 1~3단계의 기저는 무한우물의 $\psi_n$입니다.',mathjax=True),html.P('무한 퍼텐셜 우물, 조화진동자, 자유입자는 1차원 비상대론적 양자역학 모형입니다. 푸리에 변환 실험은 직교성·성분 추출·재구성을 물리계 사이에 연결합니다. Ψ는 복소 파동함수, |Ψ|²는 위치 확률밀도이며, Δx와 Δk는 확률분포의 표준편차입니다. 불확정성은 측정 기기의 부정확성을 뜻하지 않습니다.'),html.P('자유입자 아래의 벽 반사는 반직선 영역의 벽 하나를 다룹니다. 무한 퍼텐셜 우물은 양쪽 벽 사이의 별도 모형입니다. 모든 파수에서 반사 확률이 1이며 벽에서 Ψ=0입니다. 색은 복소 파동함수의 위상이며 확률을 나타내지 않습니다.')]),
    html.Details([html.Summary('수업에 활용하는 순서'),html.P('무한 퍼텐셜 우물에서 경계조건과 에너지 양자화를 탐구하고, 조화진동자의 에너지 고유상태와 중첩을 비교합니다. 파동과 분산에서는 맥놀이 → 여러 파수의 파동묶음 순서로 무분산과 물질파를 비교합니다. 자유입자에서는 평면파 → 중첩 → 푸리에 변환 → 파동묶음의 운동 → 벽 반사 순으로 탐구합니다. 푸리에 변환의 1~3단계는 무한우물의 사인 기저를 이용하고 4단계에서 자유입자의 연속 평면파 기저로 확장합니다.')]),
    html.Footer(['AI 기반 물리 시뮬레이션 교육 · 전남대학교 물리교육과 · 개발: 황정운',html.Br(),'실행 환경: Python · Dash · uv']),
    dcc.Interval(id='clock',interval=80,disabled=True),dcc.Store(id='render-busy',data=False),dcc.Store(id='browser-model'),dcc.Store(id='server-time',data=0),dcc.Store(id='time-limit',data={'end':10}),dcc.Download(id='download')],className='page')


SYSTEM_EXPERIMENTS={
    'well':[('well','에너지 고유상태 · 중첩 · 시간 변화')],
    'oscillator':[('oscillator','에너지 고유상태 · 중첩 · 시간 변화')],
    'waves':[('beat','1 맥놀이 · 군속도'),('dispersion','2 파동묶음 · 분산 비교')],
    'free':[('plane','1 평면파'),('build','2 파동의 중첩'),('fourier','3 푸리에 변환'),('free','4-1 가우스 파동묶음'),('rect','4-2 사각 파동묶음'),('wall','5 무한 벽에서의 반사')]
}

@app.callback(Output('mode','children'),Output('mode','value'),Input('system','value'))
def choose_system(system):
    experiments=SYSTEM_EXPERIMENTS[system]
    return [dcc.Tab(label=label,value=value,className='lesson-tab',selected_className='lesson-tab-selected') for value,label in experiments],experiments[0][0]

@app.callback(Output('well-controls','style'),Input('mode','value'))
def configure_well(mode):
    return {} if mode=='well' else {'display':'none'}

@app.callback(Output('well-state','data'),*[Output('well-'+key+'-value','children') for key in well.FIELDS],*[Input('well-'+key,'value') for key in well.FIELDS])
def collect_well_state(*values):
    state=dict(zip(well.FIELDS,values))
    return state,*[str(int(state[key])) if key in ('n','m') else f'{float(state[key]):.2f}' for key in well.FIELDS]

for _key in ('n','m'):
    app.clientside_callback(
        f"""function(plus,minus,value) {{
            const ctx=dash_clientside.callback_context;
            if (!ctx.triggered.length) return dash_clientside.no_update;
            const delta=ctx.triggered[0].prop_id.startsWith('well-{_key}-plus.') ? 1 : -1;
            return Math.max(1,Math.min(8,Number(value)+delta));
        }}""",
        Output('well-'+_key,'value'),Input('well-'+_key+'-plus','n_clicks'),Input('well-'+_key+'-minus','n_clicks'),State('well-'+_key,'value'),prevent_initial_call=True)

for _kind,_delta in [('weight',.1),('phase',15)]:
    for _n in range(ho.MAX_N+1):
        _id=f'ho-{_kind}-{_n}'
        _lower='Math.max(0,next)' if _kind=='weight' else 'next'
        app.clientside_callback(
            f"""function(plus,minus,value) {{
                const ctx=dash_clientside.callback_context;
                if (!ctx.triggered.length) return dash_clientside.no_update;
                const current=Number(value || 0);
                if (!Number.isFinite(current)) return dash_clientside.no_update;
                const delta=ctx.triggered[0].prop_id.startsWith('{_id}-plus.') ? {_delta} : -{_delta};
                const next=Number((current+delta).toPrecision(14));
                return {_lower};
            }}""",
            Output(_id,'value',allow_duplicate=True),Input(_id+'-plus','n_clicks'),Input(_id+'-minus','n_clicks'),State(_id,'value'),prevent_initial_call=True)


@app.callback(*[Output(f'ho-weight-{n}','value') for n in range(ho.MAX_N+1)],
              *[Output(f'ho-phase-{n}','value') for n in range(ho.MAX_N+1)],Input('ho-preset','value'),prevent_initial_call=True)
def load_oscillator_preset(preset):
    return [no_update]*(2*(ho.MAX_N+1)) if preset=='custom' else ho.preset_values(preset)


@app.callback(Output('ho-state','data'),*[Input(f'ho-weight-{n}','value') for n in range(ho.MAX_N+1)],
              *[Input(f'ho-phase-{n}','value') for n in range(ho.MAX_N+1)])
def collect_oscillator_state(*values):
    size=ho.MAX_N+1
    return dict(weights=list(values[:size]),phases=list(values[size:]))


def simulation_end(mode,k0,mass,sk,rect_state=None,beat_kind='linear'):
    if mode=='dispersion':return 24/(k0/mass)
    if mode=='beat':return 25/beat.parameters(k0,sk,mass,beat_kind)['vg']
    if mode=='rect':return rect.end_time(rect_state or rect.DEFAULT_STATE,mass)
    if mode=='free':return free_viewport(k0,mass,sk)[2]
    if mode=='wall':return wall_end_time(k0,sk,mass)
    return {'plane':10,'build':1,'oscillator':4*np.pi,'fourier':1,'well':2*np.pi}[mode]


@app.callback(Output('time-limit','data'),Input('mode','value'),Input('k0','value'),Input('mass','value'),Input('sigma-k','value'),Input('ho-state','data'),Input('well-state','data'),Input('rect-state','data'),Input('beat-kind','value'))
def configure_time_limit(mode,k0,mass,sk,harmonic_state,well_state=None,rect_state=None,beat_kind='linear'):
    return dict(end=simulation_end(mode,k0,mass,sk,rect_state,beat_kind),mode=mode,k0=k0,mass=mass,sk=sk,harmonic_state=harmonic_state,well_state=well_state,rect_state=rect_state,beat_kind=beat_kind)


# Keep the clock and reset in one synchronous browser callback. A delayed network
# tick can no longer overwrite a user's initial-condition reset or pause.
app.clientside_callback(
    """function(n, restart, mode, k0, mass, sk, ho, play, limit, t, speed, disabled, busy) {
        const ctx = dash_clientside.callback_context;
        const changed = new Set((ctx.triggered || []).map(v => v.prop_id.split('.')[0]));
        const end = mode === 'oscillator' ? 4*Math.PI :
            mode === 'free' ? Math.max(100, 30*k0/mass)/(k0/mass) :
            mode === 'plane' ? 10 : mode === 'build' ? 1 : Number((limit || {}).end || 10);
        const label = value => value ? '▶ 재생' : '❚❚ 일시정지';
        const resetKeys = ['restart','mode','k0','mass','sigma-k','ho-state','time-limit'];
        if (!ctx.triggered.length || resetKeys.some(key => changed.has(key)))
            return [0, end, true, label(true)];
        let valid = mode !== 'build' && mode !== 'fourier';
        if (mode === 'oscillator') {
            const weights=(ho || {}).weights || [];
            const phases=(ho || {}).phases || [];
            valid = weights.length === 13 && weights.every(w => Number.isFinite(Number(w || 0)) && Number(w || 0)>=0)
                && weights.some(w => Number(w || 0)>0) && phases.every(p => Number.isFinite(Number(p || 0)));
        }
        if (changed.has('play')) {
            if (valid && Number(t || 0)>=end)
                return [0, end, false, label(false)];
            const stopped = !valid || !disabled;
            return [dash_clientside.no_update, end, stopped, label(stopped)];
        }
        if (!disabled && valid && changed.has('clock')) {
            // Wait for the previous server frame instead of building a queue.
            if (busy || (typeof document !== 'undefined' && document.hidden))
                return Array(4).fill(dash_clientside.no_update);
            const next = Math.min(end, Number(t || 0)+(mode === 'well' ? .01 : .08)*Number(speed || 1));
            return [next, end, next>=end, label(next>=end)];
        }
        return [dash_clientside.no_update, end, disabled, label(disabled)];
    }""",
    Output('time','value'),Output('time','max'),Output('clock','disabled'),Output('play','children'),
    Input('clock','n_intervals'),Input('restart','n_clicks'),Input('mode','value'),Input('k0','value'),Input('mass','value'),Input('sigma-k','value'),
    Input('ho-state','data'),Input('play','n_clicks'),Input('time-limit','data'),State('time','value'),State('speed','value'),State('clock','disabled'),State('render-busy','data'))


@app.callback(Output('sigma-k-label','children'),Output('sigma-k-hint','children'),Input('mode','value'))
def configure_width_label(mode):
    if mode=='beat':return '두 파수의 차이 Δk · 단위 1/L₀','k₂=k₁+Δk. 작게 설정할수록 두 파동이 비슷해지고 포락선 간격이 넓어집니다. 가우스 파수 폭 σₖ와 다릅니다.'
    return '파수 폭 σₖ','파수들이 중심 파수 주변에 퍼져 있는 정도입니다. 가우스 기준으로 폭이 클수록 다양한 파장이 포함되고 초기 파동묶음은 좁아집니다.'


@app.callback(Output('count','value'),Output('sigma-k','value'),Input('few','n_clicks'),Input('packet','n_clicks'),Input('narrow','n_clicks'),prevent_initial_call=True)
def set_preset(_few,_packet,_narrow):
    return {'few':(7,.7),'packet':(61,.7),'narrow':(81,1.4)}[ctx.triggered_id]



@app.callback(Output('vector-intro-slot','children'),Input('mode','value'),Input('ft-stage','value'),State('vector-intro-slot','children'))
def configure_vector_intro(mode,stage,children=None):
    # Mount WebGL only for the visible lesson; removing children releases it.
    if mode!='fourier' or stage!='orthogonal':return [] if children else no_update
    return no_update if children else ft.vector_intro()


@app.callback(Output('experiment-content','className'),Input('mode','value'),Input('ft-state','data'))
def configure_plot_order(mode,fourier_state):
    if mode=='dispersion':return 'content dispersion-content'
    if mode=='beat':return 'content beat-content'
    if mode=='rect':return 'content rectangular-content'
    return 'content reconstruction-content' if mode=='fourier' and (fourier_state or ft.DEFAULT_STATE)['stage']=='reconstruct' else 'content'


@app.callback(Output('graph-equations','children'),Output('graph-equations','style'),
              Input('mode','value'),Input('count','value'),Input('sigma-k','value'),Input('ft-state','data'),Input('rect-state','data'),Input('beat-kind','value'))
def render_graph_equations(mode,count,sigma_k,fourier_state=None,rect_state=None,beat_kind='linear'):
    if mode=='dispersion':return dispersion.equations(),{}
    if mode=='beat':return beat.equations(beat_kind),{}
    if mode=='rect':return rect.equations(rect_state),{}
    if mode=='plane':
        return [math_block(r'\Psi(x,t)=A e^{i(kx-\omega t)}'),dcc.Markdown(r'$\omega=\frac{\hbar k^2}{2m}$. 그래프는 이 파동함수의 실수부·허수부·절댓값 제곱과 위상 색을 표시합니다.',mathjax=True,className='equation-note')],{}
    if mode=='fourier':
        state=fourier_state or ft.DEFAULT_STATE
        children=[ft.overview_equations(state)]
        if state['stage']=='reconstruct':
            children.insert(0,html.Div([
                html.Strong('직교성은 각 성분을 구분하게 하고, 완비성은 모든 성분을 사용하면 원래 함수를 복원할 수 있게 합니다.'),
                html.P('N을 늘리며 복원된 모양과 제곱 오차가 어떻게 달라지는지 비교해 보세요.',className='plotnote')
            ],className='completeness-intro'))
            children.append(ft.reconstruction_error_summary(state))
        return children,{}
    if mode!='build':return [],{'display':'none'}
    return [
        html.Div([
            html.Div([
                html.H4(f'N={count}개 평면파의 합',className='finite-equation-title'),
                math_block(r'\Psi_N(x,t)=\frac{\Delta k_{\mathrm{step}}}{\sqrt{2\pi}}\sum_{j=1}^{N}A(k_j)e^{i(k_jx-\omega_jt)}'),
                html.P('파란 실선: 실수부 · 분홍 실선: 절댓값 제곱',className='plotnote')
            ],className='equation-card'),
            html.Div([
                html.H4('가우스 기준 · 모든 k 포함',className='gaussian-equation-title'),
                math_block(r'\Psi_G(x,t)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}A(k)e^{i[kx-\omega(k)t]}\,dk'),
                html.P('초록 점선: 가우스 기준의 절댓값 제곱',className='plotnote')
            ],className='equation-card')
        ],className='equation-comparison'),
        dcc.Markdown(r'**현재 그래프는 두 식의 $t=0$ 값을 비교합니다.** 두 식에 같은 가우스 진폭 함수 $A(k)$를 사용합니다. $\omega(k)=\frac{k^2}{2m}$ ($\hbar=1$), $k_j=k_0+\left[j-\frac{N+1}{2}\right]\Delta k_{\mathrm{step}}$이며, '+rf'성분 간격은 $\Delta k_{{\mathrm{{step}}}}=\frac{{8\sigma_k}}{{100}}={8*sigma_k/100:.3f}$입니다. 실선과 점선 모두 같은 초기 진폭 기준 A₀로 나누어 표시합니다.',mathjax=True,className='equation-note')
    ],{}


@app.callback(Output('k0-label','children'),Output('k0-hint','children'),Output('mass-hint','children'),Input('mode','value'))
def configure_wavenumber_label(mode):
    if mode=='rect':return '중심 파수 k₀','','질량 단위 M₀ · 같은 a·k₀·시간에서 질량이 크면 중심 이동과 회절 퍼짐이 느립니다.'
    if mode=='dispersion':return '중심 파수 k₀ · 단위 1/L₀','공통 초기 파수 분포의 중심 · 무분산 속도를 물질파의 중심 속도에 자동으로 맞춥니다.','아래 물질파의 질량입니다. 두 중심을 함께 이동시키기 위해 위쪽 무분산 속도도 자동으로 맞춥니다.'
    if mode=='beat':return '첫 번째 파수 k₁ · 단위 1/L₀','두 번째 파수는 k₂=k₁+Δk입니다.','같은 두 파수에서 질량이 클수록 두 속도가 모두 느려집니다.'
    if mode=='plane':
        return '파수 k · 단위 1/L₀','하나의 평면파에 해당하는 파수 · 운동량 p=ℏk','질량 단위 M₀ · 같은 파수 k에서 질량이 크면 위상속도가 느립니다.'
    return '중심 파수 k₀ · 단위 1/L₀','초기 중심 운동량 p₀=ℏk₀','질량 단위 M₀ · 같은 k₀·σₖ·시간에서 질량이 크면 이동과 퍼짐이 느립니다.'


@app.callback(Output('options','options'),Output('options','value'),Output('play','disabled'),Output('time-box','style'),Output('presets','style'),
              Output('sigma-k-box','style'),Output('count-box','style'),Output('mass-box','style'),Output('pulse-width-box','style'),Output('ho-controls','style'),Output('k0-box','style'),Output('ft-controls','style'),Output('playback-controls','style'),Output('speed-control','style'),Input('mode','value'),Input('rect-kind','value'),Input('beat-kind','value'))
def configure(mode,rect_kind='real',beat_kind='linear'):
    labels={'real':'Re[Ψ]','imag':'Im[Ψ]','density':'|Ψ|²','components':'입사·반사 부분 함께 보기','envelope':'진폭 포락선','follow':'평균 위치 따라가기','reference':'t=0 기준 / 연속 목표'}
    if mode=='dispersion':labels.update(real='실수부',density='진폭의 절댓값 제곱',envelope='진폭 포락선',reference='초기 폭 유지 · 중심 이동 기준')
    if mode=='build':
        labels.update(real='N개 합 · Re[Ψ]',imag='N개 합 · Im[Ψ]',density='N개 합 · |Ψ|²',reference='모든 k · 가우스 밀도 비교',envelope='모든 k · 가우스 포락선')
    if mode=='plane':
        labels['phase']='위상 색 · Re[Ψ] 높이'
    if mode in ('oscillator','well','rect'):
        labels['phase']='위상 색 · |Ψ| 높이'
        labels['potential']='퍼텐셜 V · 에너지 축'
        labels['reference']='t=0 확률밀도 기준'
    if mode=='wall':
        labels['components']='입사·반사 부분 함께 보기'
        labels['phase']='위상 색 · |Ψ| 높이'
    if mode=='mirror':
        labels={'real':'합성 전기장 E','imag':'자기장 ZH','density':'에너지 밀도 u/2','components':'전체 입사파와 반사파','envelope':'입사·반사 포락선','directions':'파수의 진행 방향 표시'}
        labels.update({f'k{i+1}':html.Span(f'대표 파수 성분 {i+1}',style={'color':color}) for i,color in enumerate(COMPONENT_COLORS)})
    allowed={'dispersion':['real','density','envelope','reference'],'beat':['envelope'],'rect':['real','imag','density','phase','reference'],'plane':['real','imag','density','phase'],'build':['real','imag','density','envelope','reference'],'free':['real','imag','density','envelope','follow','reference'],'wall':['real','imag','density','phase','components'],'mirror':list(labels),'oscillator':['real','imag','density','phase','potential','reference'],'fourier':[],'well':['real','imag','density','phase','reference']}[mode]
    defaults={'dispersion':['real','density','envelope','reference'],'beat':['envelope'],'rect':['density','reference'],'plane':['real','density','phase'],'build':['real','density','reference'],'free':['real','density'],'wall':['density','phase'],'mirror':['real','directions'],'oscillator':['density','phase','potential'],'fourier':[],'well':['density','phase']}[mode]
    if mode=='rect':defaults=['real','imag','density'] if rect_kind=='boost' else ['real','density','reference']
    hide={'display':'none'};show={}
    return ([{'label':labels[k],'value':k} for k in allowed],defaults,mode in ('build','fourier'),hide if mode in ('build','fourier') else show,
            show if mode=='build' else hide,show if mode in ('build','free','wall','beat','dispersion') else hide,show if mode=='build' else hide,
            show if mode in ('plane','free','wall','rect','dispersion') or (mode=='beat' and beat_kind=='matter') else hide,show if mode=='mirror' else hide,
            show if mode=='oscillator' else hide,hide if mode in ('oscillator','fourier','well','rect') else show,
            show if mode=='fourier' else hide,hide if mode=='fourier' else show,hide if mode=='fourier' else show)


@app.callback(Output('beat-controls','style'),Input('mode','value'))
def configure_beat_controls(mode):
    return {} if mode=='beat' else {'display':'none'}


@app.callback(Output('rect-controls','style'),Input('mode','value'))
def configure_rectangle_controls(mode):
    return {} if mode=='rect' else {'display':'none'}


@app.callback(Output('rect-state','data'),Output('rect-a-value','children'),Output('rect-k0-value','children'),Output('rect-k0-control','style'),Input('rect-a','value'),Input('rect-k0','value'),Input('rect-kind','value'))
def collect_rectangle_state(a,k0,kind='real'):
    effective_k0=float(k0) if kind=='boost' else 0.
    return dict(a=float(a),k0=effective_k0,kind=kind),f'{a:.2f}',f'{effective_k0:.2f}',{} if kind=='boost' else {'display':'none'}


@app.callback(Output('ft-state','data'),*[Output('ft-'+key+'-value','children') for key in ft.FIELDS if key not in ('stage','shape')],
              *[Input('ft-'+key,'value') for key in ft.FIELDS])
def collect_fourier_state(*values):
    state=dict(zip(ft.FIELDS,values))
    labels=[str(int(state[key])) if key in ('m','n','count') else f'{float(state[key]):.2f}' for key in ft.FIELDS if key not in ('stage','shape')]
    return state,*labels


@app.callback(Output('ft-discrete-controls','style'),Output('ft-orthogonal-controls','style'),Output('ft-target-controls','style'),Output('ft-reconstruction-controls','style'),Output('ft-continuous-controls','style'),Input('ft-stage','value'))
def configure_fourier_stage(stage):
    show={};hide={'display':'none'}
    return (show if stage!='continuous' else hide,show if stage=='orthogonal' else hide,
            show if stage in ('project','reconstruct') else hide,show if stage=='reconstruct' else hide,show if stage=='continuous' else hide)


@app.callback(Output('component-plot','figure'),Output('component-title','children'),Output('component-note','children'),Output('component-panel','style'),Output('component-plot','style'),
              Input('mode','value'),Input('k0','value'),Input('sigma-k','value'),Input('count','value'),Input('pulse-width','value'),Input('server-time','data'),Input('mass','value'),Input('ft-state','data'))
def render_components(mode,k0,sk,count,sigma,t,mass,fourier_state=None):
    if mode!='wall' and set(ctx.triggered_prop_ids)=={'server-time.data'}:
        return (no_update,)*5
    changed={prop.split('.')[0] for prop in ctx.triggered_prop_ids}
    if changed & {'mode','k0','sigma-k','mass'}:t=0
    if mode=='fourier':
        fig,title,note,height=ft.detail_figure(fourier_state or ft.DEFAULT_STATE,styled_fig,line)
        return fig,title,note,{}, {'--component-height':f'{height}px','height':f'{height}px','minHeight':f'{height}px','width':'100%'}
    if mode=='wall':
        if set(ctx.triggered_prop_ids)=={'server-time.data'}:
            patch=Patch()
            fields=wall_packet(np.linspace(-8,0,241),t,k0,1/(2*sk),mass)
            a=wall_frame(k0,sk,mass,t)['amp0']
            for i,key in enumerate(['incident','reflected','psi']):
                amplitude=np.abs(fields[key])/a
                phase=np.angle(fields[key]).tolist()
                patch['data'][i]['y']=np.where(amplitude>.01,amplitude,np.nan).tolist()
                patch['data'][i]['marker']['color']=phase
                patch['data'][i]['customdata']=phase
            for i,key in enumerate(['incident','reflected']):
                ax,ay=phase_clock_arrow(fields[key][-1],a)
                patch['data'][5+i]['x']=ax.tolist()
                patch['data'][5+i]['y']=ay.tolist()
                phase=float(np.angle(fields[key][-1]))
                patch['data'][5+i]['text']=[f'벽 x=0 · 위상={phase:.3f} rad']*len(ax)
            return patch,no_update,no_update,no_update,no_update
        fig,title,note,height=wall_phase_detail(k0,sk,mass,t)
        return fig,title,note,{}, {'--component-height':f'{height}px','height':f'{height}px','minHeight':f'{height}px','width':'100%'}
    if mode!='build':return go.Figure(),'','',{'display':'none'},{'--component-height':'0px','height':'0px','minHeight':'0px'}
    if set(ctx.triggered_prop_ids)=={'server-time.data'}:
        return (no_update,)*5
    fig,title,note,height=component_figure(mode,k0,sk,count,sigma,t)
    return fig,title,note,{} if height else {'display':'none'},{'--component-height':f'{height}px','height':f'{height}px','minHeight':f'{height}px','width':'100%'}


def limit_phase_markers(figure):
    # Lines retain every sample; excessive overlapping phase dots only add SVG work.
    for trace in figure.data:
        if '색 = 위상' not in (trace.name or '') or trace.x is None:continue
        step=max(1,int(np.ceil(len(trace.x)/480)))
        trace.x=np.asarray(trace.x)[::step]
        trace.y=np.asarray(trace.y)[::step]
        trace.marker.color=np.asarray(trace.marker.color)[::step]
        if trace.customdata is not None:trace.customdata=np.asarray(trace.customdata)[::step]


def playback_position_patch(figure,move_grid=False):
    """Send changing samples and axes without resending styles or static text."""
    patch=Patch()
    # Preserve Plotly's compact binary array encoding in partial updates too.
    encoded=json.loads(figure.to_json())
    for i,trace in enumerate(encoded['data']):
        patch['data'][i]['y']=trace.get('y')
        if move_grid:patch['data'][i]['x']=trace.get('x')
        if 'customdata' in trace:patch['data'][i]['customdata']=trace['customdata']
        if 'color' in trace.get('marker',{}):patch['data'][i]['marker']['color']=trace['marker']['color']
    for key in ('shapes','annotations'):
        patch['layout'][key]=[item.to_plotly_json() for item in getattr(figure.layout,key)]
    for key in ('xaxis','yaxis'):
        patch['layout'][key]['range']=getattr(figure.layout,key).range
    return patch


PARAMS=[Input('mode','value'),Input('k0','value'),Input('sigma-k','value'),Input('count','value'),Input('mass','value'),Input('pulse-width','value'),Input('time','value'),Input('options','value'),Input('ho-state','data')]
@app.callback(Output('position','figure'),Output('spectrum','figure'),Output('metrics','children'),Output('concept','children'),Output('questions','children'),Output('status','children'),Output('position-note','children'),Output('spectrum-note','children'),Output('position-title','children'),Output('spectrum-title','children'),Output('lesson-title','children'),Output('lesson-desc','children'),Output('unit-label','children'),
              Output('k0-value','children'),Output('sigma-k-value','children'),Output('count-value','children'),Output('mass-value','children'),Output('pulse-width-value','children'),Output('time-value','children'),*[Input('server-time','data') if p.component_id=='time' else p for p in PARAMS],Input('axis-mode','value'),Input('y-zoom','value'),Input('ft-state','data'),Input('well-state','data'),Input('rect-state','data'),Input('beat-kind','value'),State('time','value'),running=[(Output('render-busy','data'),True,False)])
def render(mode,k0,sk,count,mass,sigma,t,opts,harmonic_state,axis_mode,zoom,fourier_state=None,well_state=None,rect_state=None,beat_kind='linear',actual_time=None):
    if mode in playback.LOCAL_MODES and actual_time is not None:t=actual_time
    changed={prop.split('.')[0] for prop in ctx.triggered_prop_ids}
    if changed & {'mode','k0','sigma-k','mass','ho-state','well-state','rect-state','beat-kind'}:
        t=0  # Render new initial conditions at t=0 even while the time reset is pending.
    if mode=='oscillator' and set(ctx.triggered_prop_ids)=={'server-time.data'}:
        try:
            position,metrics=ho.time_patch(harmonic_state,t,opts,axis_mode,zoom,metric)
            return (position,no_update,metrics)+(no_update,)*15+(f'{t:.2f}',)
        except (ValueError,TypeError,KeyError):
            pass  # Invalid initial conditions keep the input message visible.
    if mode=='wall' and set(ctx.triggered_prop_ids)=={'server-time.data'}:
        position,spectral,metrics=wall_frame_patches(k0,sk,mass,t,opts,axis_mode,zoom)
        return (position,spectral,metrics)+(no_update,)*15+(f'{t:.2f}',)
    if mode=='mirror' and set(ctx.triggered_prop_ids)=={'server-time.data'}:
        return (mirror_position_patch(k0,sigma,t,opts,axis_mode,zoom),)+(no_update,)*17+(f'{t:.2f}',)
    t=0 if mode=='build' else t
    result=figures(mode,k0,sk,count,mass,sigma,t,opts,axis_mode,zoom,harmonic_state,fourier_state,well_state,rect_state,beat_kind)
    limit_phase_markers(result[0])
    if mode in ('plane','free','well') and set(ctx.triggered_prop_ids)=={'server-time.data'}:
        position=playback_position_patch(result[0],move_grid=mode=='free' and 'follow' in (opts or []))
        return (position,no_update,result[2])+(no_update,)*15+(f'{t:.2f}',)
    info=MODE_INFO[mode]
    unit='전자기파 · c=1, E₀=1 · x₀=−18' if mode=='mirror' else '양자 모형 · 무차원 단위 · ℏ=1 · x₀=0'
    if mode=='dispersion':unit='파동과 분산 · 같은 초기 분포 · 같은 중심 속도 · ℏ=1'
    if mode=='beat':unit='무분산 파동 · c=V₀' if beat_kind=='linear' else '자유입자 물질파 · ℏ=1 · 속도 v₀=ℏ/(M₀L₀)'
    if mode=='plane':unit='평면파 · 길이 단위 L₀ · 질량 단위 M₀ · 시간 단위 T₀ · ℏ=1'
    if mode=='well':unit='무한우물 단위 · 길이 L · 에너지 E₁=π²ℏ²/(2mL²) · 시간 ℏ/E₁'
    if mode=='fourier':unit='함수 전개 · 1~3단계 구간 0≤x≤L · 4단계 전체 실수축 · ℏ=1'
    if mode=='oscillator':unit='조화진동자 단위 · ℏ=m=ω=1 · 길이 ℓ=√(ℏ/mω) · 시간 1/ω'
    if mode=='free':unit+=' · 연속 가우스 · 모든 k 사용'
    if mode=='rect':unit='사각 파동묶음 · 길이 L₀ · 시간 T₀=M₀L₀²/ℏ · ℏ=1'
    if mode=='wall':unit='양자 모형 · 무차원 단위 · ℏ=1 · 초기 중심 x₀=−18 · 무한 퍼텐셜 벽 x=0'
    spectrum_title={'orthogonal':'두 기저함수의 곱 · 양과 음의 상쇄','project':'목표 함수와 기저의 곱 · 계수 추출','reconstruct':'추출 계수 먼저 보기 · |cₙ| 크기순','continuous':'파수 공간의 확률밀도 |A(k)|²'}[(fourier_state or ft.DEFAULT_STATE)['stage']] if mode=='fourier' else '에너지 고유상태의 성분 확률' if mode in ('oscillator','well') else '파수 공간'
    if mode=='dispersion':spectrum_title='두 모형의 공통 파수 분포 · 시간에 무관'
    if mode=='beat':spectrum_title='분산관계 · 두 파수에서 포락선 속도 찾기'
    if mode=='rect':spectrum_title='사각 초기 상태 · 푸리에 진폭과 확률밀도'
    outputs=list((*result[:9],spectrum_title,info[1],info[2],unit,
            f'{k0:.2f}',f'{sk:.2f}',str(count),f'{mass:.2f}',f'{sigma:.2f}',f'{t:.2f}'))
    if set(ctx.triggered_prop_ids)=={'server-time.data'}:
        outputs[4]=no_update  # Keep a student's expanded explanation open during playback.
    return tuple(outputs)



@app.callback(Output('browser-model','data'),Input('mode','value'),Input('k0','value'),Input('mass','value'),Input('sigma-k','value'),Input('options','value'),Input('axis-mode','value'),Input('y-zoom','value'),Input('ho-state','data'),Input('well-state','data'),Input('rect-state','data'),Input('beat-kind','value'))
def configure_browser_model(mode,k,mass,sk,options,axis,zoom,harmonic,well_state,rect_state=None,beat_kind='linear'):
    return playback.model(mode,k,mass,sk,options,axis,zoom,harmonic,well_state,rect_state,beat_kind)


app.clientside_callback(
    "function(t,mode){return ['plane','beat','dispersion','free','rect','well','oscillator','build','fourier'].includes(mode)?dash_clientside.no_update:t;}",
    Output('server-time','data'),Input('time','value'),Input('mode','value'))

app.clientside_callback(
    "function(t,model,figure,mode){return window.quantumPlayback.render(t,model,figure,mode);}",
    Output('position','figure',allow_duplicate=True),Output('metrics','children',allow_duplicate=True),Output('time-value','children',allow_duplicate=True),
    Input('time','value'),Input('browser-model','data'),State('position','figure'),State('mode','value'),prevent_initial_call=True)


@app.callback(Output('download','data'),Input('export','n_clicks'),*[State(p.component_id,p.component_property) for p in PARAMS],State('ft-state','data'),State('well-state','data'),State('rect-state','data'),State('beat-kind','value'),prevent_initial_call=True)
def export_data(_n,mode,k0,sk,count,mass,sigma,t,opts,harmonic_state,fourier_state=None,well_state=None,rect_state=None,beat_kind='linear'):
    t=0 if mode=='build' else t
    data=figures(mode,k0,sk,count,mass,sigma,t,opts,harmonic_state=harmonic_state,fourier_state=fourier_state,well_state=well_state,rect_state=rect_state,beat_kind=beat_kind)[-1]
    names=list(data)
    header=f'# mode={mode},t={t},k0={k0},sigma_k={sk},N={count},m={mass},pulse_sigma={sigma}\n'
    if mode=='oscillator':header=f'# mode=oscillator,t={t},hbar=m=omega=1\n# weights={harmonic_state["weights"]}\n# phase_degrees={harmonic_state["phases"]}\n'
    if mode=='well':header=f'# mode=well,time_E1_over_hbar={t}, settings={well_state}\n'
    if mode=='fourier':header=f'# mode=fourier, settings={fourier_state}\n'
    if mode=='beat':header=f'# mode=beat,dispersion={beat_kind},t={t},k1={k0},dk={sk},mass={mass},c=1 (linear only)\n'
    if mode=='rect':header=f'# mode=rect,t={t},mass={mass},settings={rect_state},A0=1/sqrt(2a)\n'
    rows=[','.join(names)]
    rows+= [','.join(f'{float(v):.10g}' for v in row) for row in zip(*(data[key] for key in names))]
    return dict(content=header+'\n'.join(rows),filename=f'wave-{mode}-t{t:.2f}.csv',type='text/csv')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Quantum mechanics classroom simulator')
    parser.add_argument('--port',type=int,default=8050)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--debug',action='store_true')
    args=parser.parse_args()
    app.run(host=args.host,port=args.port,debug=args.debug)
