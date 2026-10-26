"""Orthogonality, projection, synthesis, and the continuous Fourier transform."""
from functools import lru_cache
import numpy as np
import plotly.graph_objects as go
from dash import html,dcc
from physics import sine_basis,gaussian_packet,gaussian_fourier_amplitude
from teaching_content import question_items,math_block

DEFAULT_STATE=dict(stage='orthogonal',length=10.,m=2,n=1,count=1,shape='gaussian',width=.1,sigma_x=1.,x0=0.,k0=0.)
FIELDS=['stage','length','m','n','count','shape','width','sigma_x','x0','k0']



def control_slider(key,title,lo,hi,step,value):
    return html.Div([html.Div([html.Label(title),html.Output(id='ft-'+key+'-value',className='value')],className='labelrow'),
        dcc.Slider(id='ft-'+key,min=lo,max=hi,step=step,value=value,marks=None)],className='control')


def controls():
    return html.Div([
        html.H3('분해하고 다시 합성하기'),
        dcc.RadioItems(id='ft-stage',options=[{'label':label,'value':key} for key,label in [('orthogonal','1. 무한우물 기저 · 직교성'),('project','2. 무한우물 기저 · 계수 추출'),('reconstruct','3. 무한우물 기저 · 재구성'),('continuous','4. 자유입자 · 연속 푸리에 변환')]],value='orthogonal',className='ft-stages'),
        html.Div([control_slider('length','구간 길이 L',4,20,1,10),control_slider('n','선택한 성분 n',1,40,1,1)],id='ft-discrete-controls'),
        html.Div(control_slider('m','비교할 성분 m',1,40,1,2),id='ft-orthogonal-controls'),
        html.Div([html.Label('목표 함수 선택 · f(x)',className='ft-shape-heading'),html.P('아래 항목을 눌러 목표 함수를 바꿔 보세요.',className='hint'),dcc.RadioItems(id='ft-shape',options=[{'label':label,'value':key} for key,label in [('gaussian','가우스형 봉우리'),('double','비대칭 두 봉우리'),('triangle','삼각형'),('offcenter','왼쪽으로 치우친 봉우리'),('rectangle','직사각형 · 불연속 예시')]],value='gaussian',className='ft-shapes'),
            control_slider('width','봉우리 폭 설정 η',.05,.25,.01,.1),html.P('가우스형 봉우리로 시작하고, 비대칭 두 봉우리로 더 복잡한 함수의 계수와 복원을 비교하세요. 직사각형은 불연속점의 잔물결 예시입니다.',className='hint')],id='ft-target-controls'),
        html.Div(control_slider('count','큰 계수부터 포함할 성분 수 N',1,60,1,1),id='ft-reconstruction-controls'),
        html.Div([control_slider('sigma_x','위치 확률분포의 표준편차 σₓ',.3,2,.1,1),control_slider('x0','중심 위치 x₀',-4,4,.2,0),control_slider('k0','중심 파수 k₀',-4,4,.2,0)],id='ft-continuous-controls'),
        dcc.Store(id='ft-state',data=DEFAULT_STATE)
    ],id='ft-controls',style={'display':'none'})


def cumulative_integral(x,y):
    return np.concatenate([[0.],np.cumsum((y[1:]+y[:-1])*.5*np.diff(x))])


@lru_cache(maxsize=4)
def discrete_basis(length):
    x=np.linspace(0,length,4097)
    return x,sine_basis(x,np.arange(1,61),length)


@lru_cache(maxsize=8)
def discrete_data(length,shape,width):
    x,basis=discrete_basis(length);z=x/length
    if shape in ('gaussian','offcenter'):
        center=.5 if shape=='gaussian' else .3
        f=np.sin(np.pi*z)*np.exp(-(z-center)**2/(2*width**2))
    elif shape=='double':
        narrow=np.exp(-(z-.28)**2/(2*(.55*width)**2))
        wide=.65*np.exp(-(z-.70)**2/(2*width**2))
        f=np.sin(np.pi*z)*(narrow+wide)
    elif shape=='triangle':f=np.maximum(1-np.abs(z-.5)/(2*width),0)
    elif shape=='rectangle':f=(np.abs(z-.5)<width).astype(float)
    else:raise ValueError('unknown target function')
    f/=np.sqrt(np.trapezoid(f*f,x))
    coefficients=np.trapezoid(basis*f,x,axis=1)
    return x,f,coefficients,basis


def product_plot(x,y,title,styled_fig):
    fig=styled_fig(title,'위치 x','곱의 값',360)
    for values,name,color in [(np.maximum(y,0),'양의 영역','#74e0aa'),(np.minimum(y,0),'음의 영역','#ff7daf')]:
        fig.add_trace(go.Scatter(x=x,y=values,name=name,mode='lines',fill='tozeroy',line=dict(color=color,width=1.5)))
    fig.add_hline(y=0,line_color='white',line_width=1)
    fig.update_xaxes(range=[float(x[0]),float(x[-1])])
    return fig


def ranked_reconstruction(coefficients,basis,count):
    order=np.argsort(-np.abs(coefficients),kind='stable')
    selected=order[:int(count)]
    return coefficients[selected]@basis[selected],selected,order


def reconstruction_error_summary(state):
    _,_,c,basis=discrete_data(float(state['length']),state['shape'],float(state['width']))
    _,selected,_=ranked_reconstruction(c,basis,state['count'])
    remaining=max(0.,float(1-np.sum(np.abs(c[selected])**2)))
    value=f'{remaining:.6f}' if remaining>=1e-6 or remaining==0 else f'{remaining:.3e}'
    if 'e' in value:
        mantissa,exponent=value.split('e');value=mantissa+r'\times10^{'+str(int(exponent))+'}'
    return html.Div([
        html.H4('계수로 계산한 제곱 오차'),
        math_block(r'1-\sum_{r=1}^{N}|c_{n_r}|^2='+value),
        dcc.Markdown(r'정규화한 목표 함수에서는 이 값이 $\int_0^L|f-f_N|^2\,dx$와 같습니다. 현재 포함한 성분만 합산하며 **0에 가까울수록 재구성이 더 정확합니다.**',mathjax=True,className='equation-note')
    ],className='reconstruction-error-summary')


def overlap_data(state):
    x,f,coefficients,basis=discrete_data(float(state['length']),state['shape'],float(state['width']))
    n=int(state['n']);m=int(state['m'])
    product=basis[m-1]*basis[n-1] if state['stage']=='orthogonal' else f*basis[n-1]
    return x,f,coefficients,basis,product


def overview_equations(state):
    stage=(state or DEFAULT_STATE)['stage']
    if stage=='orthogonal':
        return math_block(r'\psi_n(x)=\sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)',r'\int_0^L\psi_m^*(x)\psi_n(x)\,dx=\delta_{mn}')
    if stage=='project':
        return math_block(r'f(x)=\sum_jc_j\psi_j(x)',r'c_n=\int_0^L\psi_n^*(x)f(x)\,dx')
    if stage=='reconstruct':
        return math_block(r'f_N(x)=\sum_{r=1}^N c_{n_r}\psi_{n_r}(x)',r'|c_{n_1}|\geq|c_{n_2}|\geq\cdots',r'\int_0^L|f-f_N|^2\,dx=1-\sum_{r=1}^N|c_{n_r}|^2')
    return math_block(r'A(k)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\Psi(x,0)e^{-ikx}\,dx',r'\Psi(x,0)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}A(k)e^{ikx}\,dk')


def figures(state,axis_mode,zoom,styled_fig,line,metric,fit_axis):
    state=state or DEFAULT_STATE;stage=state['stage']
    position=styled_fig('fourier-'+stage,'위치 x','함수의 값',540)
    spectrum=styled_fig('fourier-detail-'+stage,'위치 x','곱의 값',360)
    base=[html.P('기저함수는 함수를 표현할 때 사용하는 성분 함수입니다. 직교성은 전체 구간의 내적이 0이라는 뜻입니다. 같은 위치에서 두 함수가 함께 존재해도 곱의 적분은 상쇄될 수 있습니다. 1~3단계의 이산 정규직교 기저에서는 자기 자신과의 내적이 1입니다.'),
          html.P('복소 함수의 내적은 ∫ψₙ*(x)f(x)dx입니다. 1~3단계의 사인 기저는 실수 함수이므로 켤레를 붙여도 값이 같습니다.'),
          html.P('기저함수 전개는 에너지 고유상태의 중첩과 같은 구조입니다. 무한우물에서는 경계조건으로 이산적인 사인 모드가 정해지고, 전체 자유공간에서는 연속 파수의 평면파를 사용합니다. 구간을 넓힐 때 인접 사인 모드의 파수 간격 π/L는 작아지지만, 사인 모드의 합을 그대로 전체 실수축의 푸리에 변환과 동일시하지 않습니다.'),
          html.P([html.A('참고: MIT · Fourier series and integrals',href='https://math.mit.edu/~gs/cse/websections/cse41.pdf',target='_blank')],className='mini')]
    warning=''
    if stage=='continuous':
        sx=float(state['sigma_x']);x0=float(state['x0']);k0=float(state['k0'])
        x=np.linspace(-12,12,2401);k=np.linspace(-12,12,2401)
        psi=gaussian_packet(x,0,k0,sx,x0=x0);a=gaussian_fourier_amplitude(k,k0,sx,x0)
        line(position,x,np.abs(psi)**2,'|Ψ(x)|²','#ff7daf',fill=True)
        line(spectrum,k,np.abs(a)**2,'|A(k)|²','#ffd166',fill=True)
        position.update_xaxes(range=[-12,12]);spectrum.update_xaxes(range=[-12,12],title='파수 k')
        position.update_yaxes(range=[0,1/(np.sqrt(2*np.pi)*.3)*1.12],title='위치 확률밀도')
        spectrum.update_yaxes(range=[0,1/(np.sqrt(2*np.pi)*.25)*1.12],title='파수 확률밀도')
        position.add_vline(x=x0,line_color='white',line_dash='dash');spectrum.add_vline(x=k0,line_color='white',line_dash='dash')
        metrics=[metric('위치 폭 Δx',f'{sx:.3f}'),metric('파수 폭 Δk',f'{1/(2*sx):.3f}'),metric('Δx·Δp / ℏ','0.50000'),metric('중심 위치 / 중심 파수',f'{x0:.1f} / {k0:.1f}')]
        free_concepts=[
            dcc.Markdown(r'**4단계는 자유입자($V=0$, 전체 실수축)의 가우스 초기 상태를 연속 평면파로 분해하는 실험입니다.** 1~3단계는 무한우물의 이산 사인 기저로 직교성·성분 추출·재구성을 익히는 준비 단계입니다. 푸리에 변환 자체는 자유입자에만 적용되는 기법은 아닙니다.',mathjax=True),
            math_block(r'\psi_k(x)=\frac{e^{ikx}}{\sqrt{2\pi}},\quad E(k)=\frac{\hbar^2k^2}{2m}',r'A(k)=\langle\psi_k|\Psi(0)\rangle'),
            dcc.Markdown(r"평면파 기저는 연속 파수로 구분되며 $\langle\psi_{k'}|\psi_k\rangle=\delta(k-k')$로 규격화합니다. 무한한 공간의 평면파 하나는 제곱 적분이 1인 상태가 아니며, 이를 중첩한 가우스 파동묶음은 정규화할 수 있습니다.",mathjax=True),
            dcc.Markdown(r'**변환하는 것은 복소 파동함수 $\Psi(x,0)$입니다.** 위 두 그래프는 변환한 다음 계산한 $|\Psi(x,0)|^2$와 $|A(k)|^2$입니다. 위치 확률밀도를 직접 푸리에 변환한 결과가 파수 확률밀도인 것은 아닙니다. 아래에서 계수의 실수부·허수부와 위상 정보를 확인할 수 있습니다.',mathjax=True),
            math_block(r'\Psi(x,0)=\frac{e^{-\frac{(x-x_0)^2}{4\sigma_x^2}}e^{ik_0(x-x_0)}}{(2\pi\sigma_x^2)^{1/4}}',r'A(k)=\left(\frac{2\sigma_x^2}{\pi}\right)^{1/4}e^{-\sigma_x^2(k-k_0)^2}e^{-ikx_0}'),
            math_block(r'A(k,t)=A(k,0)e^{-iE(k)t/\hbar}'),
            dcc.Markdown(r'이 화면은 $t=0$의 변환쌍을 비교합니다. 자유입자의 시간 변화에서는 각 파수 성분의 위상만 위 식처럼 변하므로 $|A(k,t)|^2$는 일정하지만 위치 분포는 이동하고 퍼집니다.',mathjax=True)
        ]
        concepts=[
            dcc.Markdown('실수 양의 가우스 포락선에 중심 파수의 위상을 곱한 t=0 상태입니다. σₓ는 |Ψ|²의 표준편차이며 $\\Delta k=\\frac{1}{2\\sigma_x}$, Δp=ℏΔk입니다.',mathjax=True),
            dcc.Markdown('x₀를 바꾸면 위치 분포가 이동하고 A(k)에 $e^{-ikx_0}$가 곱해집니다. 따라서 |A(k)|²는 바뀌지 않지만 복소 계수의 위상은 변합니다. k₀를 바꾸면 파수 분포의 중심이 이동합니다.',mathjax=True),*base]
        concepts=free_concepts+concepts
        note='전체 실수축의 정규화된 가우스와 그 정확한 푸리에 변환을 사용합니다. 화면은 −12~12만 표시하며 화면 밖의 꼬리를 잘라 재정규화하지 않습니다. 시간 변화는 자유입자의 「파동묶음의 운동」에서 확인합니다.'
        knote='이 탭은 t=0의 최소 불확정도 가우스입니다. 폭을 바꾸면 Δx·Δp/ℏ=0.5를 유지합니다. 자유입자 탭의 시간에 따른 퍼짐에서는 이 곱이 증가합니다.'
        title='자유입자의 푸리에 변환 · 위치와 파수의 연결'
        data=dict(x=x,real=psi.real,imag=psi.imag,density=np.abs(psi)**2)
    else:
        length=float(state['length']);n=int(state['n']);m=int(state['m']);count=int(state['count'])
        x,f,c,basis,product=overlap_data(state)
        positive=float(np.trapezoid(np.maximum(product,0),x));negative=float(np.trapezoid(np.minimum(product,0),x));total=positive+negative
        total=0. if abs(total)<1e-12 else total
        concepts=[math_block('\\psi_n(x)=\\sqrt{\\frac{2}{L}}\\sin\\left(\\frac{n\\pi x}{L}\\right),\\quad n=1,2,\\ldots','\\int_0^L \\psi_m^*(x)\\psi_n(x)\\,dx=\\delta_{mn}'),*base]
        if stage=='orthogonal':
            concepts=base.copy()
            line(position,x,basis[m-1],f'ψₘ · m={m}','#48d7ff');line(position,x,basis[n-1],f'ψₙ · n={n}','#ffd166',dash='dash')
            spectrum=product_plot(x,product,'fourier-orthogonal-product',styled_fig)
            metrics=[metric('양의 곱 적분',f'{positive:.6f}'),metric('음의 곱 적분',f'{negative:.6f}'),metric('내적 Iₘₙ',f'{total:.6f}'),metric('예상값 δₘₙ','1' if m==n else '0')]
            note='두 모드는 같은 구간 [0,L]에서 규격화했습니다. m=n이면 동일한 함수 두 개가 겹칩니다.'
            knote='초록색과 분홍색은 곱의 부호입니다. m≠n일 때 양·음의 적분이 상쇄됩니다. m=n일 때 곱은 음수가 아니며 전체 적분은 1입니다.'
            title='직교성 · 곱해서 적분하면 겹침을 측정합니다'
        elif stage=='project':
            line(position,x,f,'목표 함수 f(x)','#ff7daf');line(position,x,basis[n-1],f'선택한 기저 ψₙ · n={n}','#48d7ff',dash='dash')
            spectrum=product_plot(x,product,'fourier-projection-product',styled_fig)
            metrics=[metric('양의 곱 적분',f'{positive:.6f}'),metric('음의 곱 적분',f'{negative:.6f}'),metric(f'추출 계수 c{n}',f'{c[n-1]:.6f}'),metric('정규화 ∫|f|²dx','1.000000')]
            note='f와 ψₙ을 같은 구간에서 비교합니다. 목표 함수는 제곱 적분이 1이 되도록 정규화했습니다. 가우스형 봉우리는 sin(πx/L)×exp[−(x/L−a)²/(2η²)]이며 a=0.5, 치우친 봉우리는 a=0.3입니다. 사인 인자로 양끝을 0으로 맞췄습니다.'
            knote='cₙ은 부호를 갖는 진폭 계수입니다. 음수는 음의 확률이 아니라 해당 성분의 부호 반전을 뜻합니다. 복소 기저에서는 일반적으로 계수도 복소수입니다.'
            title='계수 추출 · 기저함수에 대한 투영'
        else:
            partial,selected,order=ranked_reconstruction(c,basis,count)
            line(position,x,f,'목표 함수 f(x)','#ff7daf');line(position,x,partial,f'부분합 f_N · N={count}','#48d7ff',dash='dash')
            ranks=np.arange(1,len(c)+1);indices=order+1
            spectrum=styled_fig('fourier-ranked-coefficients','추가 순서 r'+' · |cₙ|이 큰 순서','계수 cₙ · 부호 유지',360)
            colors=['#ff7daf' if v<0 else '#ffd166' for v in c[order]]
            spectrum.add_trace(go.Bar(x=ranks,y=c[order],name='크기순으로 정렬한 추출 계수',marker=dict(color=colors,opacity=np.where(ranks<=count,1,.25)),customdata=np.column_stack([indices,indices*np.pi/length,np.abs(c[order])]),hovertemplate='추가 순서=%{x}<br>n=%{customdata[0]:.0f}<br>kₙ=%{customdata[1]:.3f}<br>cₙ=%{y:.5f}<br>|cₙ|=%{customdata[2]:.5f}<extra></extra>'))
            chosen_rank=int(np.flatnonzero(order==n-1)[0])+1
            spectrum.add_trace(go.Scatter(x=[chosen_rank],y=[c[n-1]],mode='markers',marker=dict(color='#48d7ff',size=11,symbol='circle-open'),name=f'선택한 n={n} · 순서 {chosen_rank}',customdata=[[n,n*np.pi/length,abs(c[n-1])]],hovertemplate='추가 순서=%{x}<br>n=%{customdata[0]:.0f}<br>kₙ=%{customdata[1]:.3f}<br>cₙ=%{y:.5f}<br>|cₙ|=%{customdata[2]:.5f}<extra></extra>'))
            spectrum.update_layout(hovermode='closest',hoverdistance=30)
            spectrum.update_xaxes(range=[.5,60.5]);spectrum.add_hline(y=0,line_color='white')
            error=float(np.trapezoid((f-partial)**2,x));power=float(np.sum(c[selected]**2))
            metrics=[metric('포함한 성분 수 N',str(count)),metric('포함한 제곱 계수 합',f'{power:.6f}'),metric('오차 ∫|f−f_N|²dx',f'{error:.6f}'),metric('계수 오차 1−Σ|cₙ|²',f'{max(0.,1-power):.6f}')]
            note='|cₙ|이 큰 성분부터 N개를 더한 결과입니다. N은 최대 양자수가 아니라 포함한 성분의 개수입니다. 계수의 부호는 그대로 사용합니다. 부분합을 다시 정규화하지 않습니다. 삼각형은 max[1−|x/L−0.5|/(2η),0], 직사각형은 |x/L−0.5|<η에서 일정한 값입니다. 두 함수도 제곱 적분을 1로 정규화합니다.'
            knote='먼저 1~60의 추출 계수를 |cₙ|이 큰 순서로 정렬해 보여줍니다. 가로축은 양자수 n이 아니라 추가 순서이며 실제 n은 마우스를 올리면 확인할 수 있습니다. 진한 막대는 부분합에 포함한 성분, 옅은 막대는 아직 제외한 성분입니다. 음의 계수는 부호 반전입니다. 선택한 n의 원은 아래 성분 그래프와 연결됩니다. N이 커져도 모든 지점에서 오차가 단조 감소하는 것은 아닙니다.'
            sequence=', '.join(str(int(i)+1) for i in order[:min(count,10)])
            knote+=f' 현재 포함 순서: n={sequence}'+(' …' if count>10 else '')+'.'
            title='큰 계수부터 재구성 · f(x)에 얼마나 가까워질까?'
            concepts.extend([
                html.H4('완비성 · 무한차원 함수 공간과 수렴'),
                html.P('직교성만으로 완비성이 보장되지는 않습니다. 이 함수 공간은 무한차원이므로 공간 전체를 표현하려면 무한히 많은 기저함수가 필요합니다. 특정 함수는 몇 개 성분만으로 정확히 표현될 수도 있습니다.'),
                dcc.Markdown(r'무한우물의 사인 기저 $\psi_n(x)$ ($n=1,2,\ldots$)는 구간 $[0,L]$의 **제곱 적분 가능한 함수 공간**에서 완비입니다. 따라서 모든 성분을 포함하는 극한에서 부분합과 원래 함수의 제곱 오차 적분이 0으로 갑니다.',mathjax=True),
                math_block(r'\lim_{N\to\infty}\int_0^L|f(x)-f_N(x)|^2\,dx=0'),
                html.P('이는 모든 점에서 오차가 같은 속도로 줄거나 매번 줄어든다는 뜻이 아닙니다. 완비성의 극한은 빠짐없이 무한한 기저를 포함하는 경우입니다. 이 화면은 먼저 n=1~60의 후보 안에서 |cₙ|이 큰 순서로 선택하는 유한 근사이므로, N=60 뒤에도 n>60의 누락 성분에 따른 오차가 남을 수 있습니다.'),
                html.P([html.A('참고: MIT · Fourier series의 완비성과 L² 수렴',href='https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/859833f65ee4aec6b0fea35a300d10cd_MIT18_102s21_lec15.pdf',target='_blank')],className='mini')
            ])

            if state['shape']=='rectangle':warning='직사각형의 불연속점 근처에는 유한한 성분으로 근사하는 과정의 잔물결이 나타날 수 있습니다. 표준 Gibbs 설명은 낮은 파수부터 더하는 순서에 대한 것입니다. 여기서는 |cₙ| 크기순으로 선택하므로 N 증가에 따른 국소적인 잔물결 변화와 전체 제곱 오차 감소를 구분하세요.'
        if state['shape']=='double' and stage in ('project','reconstruct'):
            concepts.extend([
                html.H4('이번 목표 함수 · 비대칭 두 봉우리'),
                math_block(r'\begin{aligned}f(x)=C\sin(\pi z)\bigg[&e^{-\frac{(z-0.28)^2}{2(0.55\eta)^2}}\\ &+0.65e^{-\frac{(z-0.70)^2}{2\eta^2}}\bigg]\end{aligned}',r'z=\frac{x}{L},\qquad\int_0^L|f(x)|^2\,dx=1'),
                html.P('왼쪽에는 좁은 봉우리, 오른쪽에는 더 넓고 낮은 봉우리를 넣었습니다. C는 정규화 상수이며 사인 인자로 양끝을 0으로 맞춥니다. η를 크게 하면 봉우리가 넓어져 서로 겹칠 수 있습니다. 단일 봉우리와 비교해 어떤 n의 계수가 커지는지, 같은 N으로 두 봉우리를 얼마나 복원하는지 관찰하세요.')
            ])
        position.update_xaxes(range=[0,length]);position.update_yaxes(range=[-max(1,np.max(np.abs(f))*1.2),max(1,np.max(np.abs(f))*1.2)])
        if stage!='reconstruct':
            edge=max(abs(product).max()*1.15,1e-8);spectrum.update_yaxes(range=[-edge,edge])
        data=dict(x=x,target=f,basis=basis[n-1],product=product)
        if stage=='orthogonal':data=dict(x=x,basis_m=basis[m-1],basis_n=basis[n-1],product=product,cumulative=cumulative_integral(x,product))
        elif stage=='project':data['cumulative']=cumulative_integral(x,product)
        if stage=='reconstruct':data['partial_sum']=partial
    fit_axis(position,axis_mode,zoom);fit_axis(spectrum,'fixed' if stage=='continuous' else 'auto',1)
    return position,spectrum,metrics,concepts,question_items('fourier-'+stage,0),warning,note,knote,title,data


def detail_figure(state,styled_fig,line):
    state=state or DEFAULT_STATE;stage=state['stage']
    if stage=='continuous':
        k=np.linspace(-12,12,2401)
        a=gaussian_fourier_amplitude(k,state['k0'],state['sigma_x'],state['x0'])
        fig=styled_fig('fourier-amplitude','파수 k','계수 A(k)의 실수부·허수부',360)
        line(fig,k,a.real,'Re A(k)','#48d7ff');line(fig,k,a.imag,'Im A(k)','#ff7daf',dash='dash')
        fig.update_xaxes(range=[-12,12]);fig.update_yaxes(range=[-1.4,1.4])
        return fig,'복소 계수 보기 · x₀를 바꾸면 무엇이 변할까?','x₀를 바꾸어도 |A(k)|²는 같지만 A(k)의 실수부·허수부는 바뀝니다. 재구성에는 계수의 크기와 위상 정보가 모두 필요합니다.',360
    x,f,c,basis,product=overlap_data(state);n=int(state['n'])
    if stage=='reconstruct':
        fig=styled_fig('fourier-one-mode','위치 x','개별 성분의 값',360)
        line(fig,x,basis[n-1],f'기저 ψₙ · n={n}','#a8bad0',dash='dash',width=1.5)
        line(fig,x,c[n-1]*basis[n-1],f'기여 cₙψₙ · cₙ={c[n-1]:.5f}','#48d7ff')
        fig.update_xaxes(range=[0,state['length']])
        _,selected,order=ranked_reconstruction(c,basis,state['count'])
        rank=int(np.flatnonzero(order==n-1)[0])+1
        included='포함됨' if n-1 in selected else '아직 제외됨'
        return fig,f'선택한 n={n}의 기여 · 추가 순서 {rank} · {included}','회색은 기저함수, 파란색은 추출한 계수를 곱한 성분입니다. 포함 여부는 n≤N이 아니라 |cₙ| 크기순으로 정한 추가 순서에 따라 결정됩니다.',360
    integral=cumulative_integral(x,product)
    fig=styled_fig('fourier-cumulative','적분 상한 a','누적 적분 ∫₀ᵃ 곱 dx',360)
    line(fig,x,integral,'누적 적분','#ffd166')
    fig.add_hline(y=0,line_color='white');fig.add_vline(x=state['length'],line_color='white',line_dash='dash')
    fig.update_xaxes(range=[0,state['length']])
    return fig,'누적 적분' if stage=='orthogonal' else '누적 적분 · 계수가 만들어지는 과정','곡선은 적분 상한을 0에서 L까지 옮기며 곱을 누적한 값입니다. 최종값은 전체 구간의 내적입니다. 재생 애니메이션이 아니라 적분 상한에 따른 그래프입니다.',360


def vector_intro():
    """A geometric bridge from coordinate projections to function coefficients."""
    vector=np.array([3.,2.,2.])
    colors=['#ffd166','#ff7daf','#74e0aa'];labels=['x','y','z']
    fig=go.Figure()
    for axis,(color,label) in enumerate(zip(colors,labels)):
        direction=np.eye(3)[axis];end=direction*4
        fig.add_trace(go.Scatter3d(x=[0,end[0]],y=[0,end[1]],z=[0,end[2]],mode='lines',line=dict(color='#496078',width=3),showlegend=False,hoverinfo='skip'))
        unit=direction
        fig.add_trace(go.Cone(x=[unit[0]],y=[unit[1]],z=[unit[2]],u=[unit[0]],v=[unit[1]],w=[unit[2]],anchor='tip',sizemode='absolute',sizeref=.24,colorscale=[[0,color],[1,color]],showscale=False,showlegend=False,hoverinfo='skip'))
        projection=direction*vector[axis]
        fig.add_trace(go.Scatter3d(x=[0,projection[0]],y=[0,projection[1]],z=[0,projection[2]],mode='lines+markers',name=f'{label} 성분 = {vector[axis]:g}',line=dict(color=color,width=7),marker=dict(size=4,color=color),hovertemplate=f'{label} 방향 성분 = {vector[axis]:g}<extra></extra>'))
        fig.add_trace(go.Scatter3d(x=[vector[0],projection[0]],y=[vector[1],projection[1]],z=[vector[2],projection[2]],mode='lines',line=dict(color=color,width=5,dash='dash'),opacity=.9,showlegend=False,hoverinfo='skip'))
        text=direction*4.15
        fig.add_trace(go.Scatter3d(x=[text[0]],y=[text[1]],z=[text[2]],mode='text',text=[f'{label} · e{["ₓ","ᵧ","𝓏"][axis]}'],textfont=dict(color=color,size=18),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[0,vector[0]],y=[0,vector[1]],z=[0,vector[2]],mode='lines',name='벡터 v = (3, 2, 2)',line=dict(color='#48d7ff',width=8),hovertemplate='v = (3, 2, 2)<extra></extra>'))
    fig.add_trace(go.Cone(x=[vector[0]],y=[vector[1]],z=[vector[2]],u=[vector[0]],v=[vector[1]],w=[vector[2]],anchor='tip',sizemode='absolute',sizeref=.35,colorscale=[[0,'#48d7ff'],[1,'#48d7ff']],showscale=False,showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[vector[0]],y=[vector[1]],z=[vector[2]+.25],mode='text',text=['v = (3, 2, 2)'],textfont=dict(color='#48d7ff',size=20),showlegend=False,hoverinfo='skip'))
    axis_style=dict(range=[-.4,4.6],showbackground=False,showgrid=False,zeroline=False,showticklabels=False,title='')
    fig.update_layout(template='plotly_dark',height=450,paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=55,b=0),font=dict(size=16,color='#f4f8ff'),legend=dict(orientation='h',x=0,y=1.04),hoverlabel=dict(bgcolor='#10223a',font=dict(color='#f4f8ff',size=16)),scene=dict(xaxis=axis_style,yaxis=axis_style,zaxis=axis_style,aspectmode='cube',camera=dict(eye=dict(x=1.2,y=-1.5,z=.95),projection=dict(type='orthographic'))),uirevision='vector-intro')
    views=[
        ('입체 시점',dict(x=1.2,y=-1.5,z=.95),dict(x=0,y=0,z=1)),
        ('x-y 평면',dict(x=0,y=0,z=2.2),dict(x=0,y=1,z=0)),
        ('y-z 평면',dict(x=2.2,y=0,z=0),dict(x=0,y=0,z=1)),
        ('x-z 평면',dict(x=0,y=-2.2,z=0),dict(x=0,y=0,z=1))
    ]
    buttons=[dict(label=label,method='relayout',args=[{'scene.camera':dict(eye=eye,up=up,center=dict(x=0,y=0,z=0),projection=dict(type='orthographic'))}]) for label,eye,up in views]
    fig.update_layout(margin=dict(l=0,r=0,t=130,b=0),updatemenus=[dict(type='dropdown',direction='down',x=1,y=1.24,xanchor='right',yanchor='top',showactive=True,active=0,buttons=buttons,bgcolor='#10223a',bordercolor='#496078',font=dict(color='#f4f8ff',size=16))])
    return html.Article([
        html.H3('먼저, 벡터의 성분을 내적으로 꺼내 보기'),
        html.P('서로 수직인 세 단위 방향벡터 중 하나와 내적하면 그 방향의 좌표 성분만 남습니다. 오른쪽 위 메뉴에서 입체·평면 시점을 선택하거나, 그림을 드래그해 돌려 볼 수 있습니다. 시점을 바꾸어도 벡터와 성분 값은 변하지 않습니다.',className='plotnote'),
        dcc.Graph(id='vector-intro-graph',figure=fig,responsive=True,className='vector-intro-graph',config={'displaylogo':False,'modeBarButtonsToRemove':['select2d','lasso2d']}),
        math_block(r'\mathbf v=3\mathbf e_x+2\mathbf e_y+2\mathbf e_z',r'\mathbf e_x\cdot\mathbf v=3,\quad\mathbf e_y\cdot\mathbf v=2,\quad\mathbf e_z\cdot\mathbf v=2',r'\mathbf e_i\cdot\mathbf e_j=\delta_{ij}'),
        dcc.Markdown(r'**함수도 같은 방식입니다.** 벡터의 방향 성분을 내적으로 꺼내듯이, 함수에서는 기저함수와의 내적으로 계수를 꺼냅니다. $c_n=\langle\psi_n\mid f\rangle=\int_0^L\psi_n^*(x)f(x)\,dx$. 복소 함수에서는 기저의 복소켤레를 사용합니다.',mathjax=True)
    ],id='vector-intro-panel',className='plot')
