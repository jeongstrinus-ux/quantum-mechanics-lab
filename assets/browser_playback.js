// Exact analytic evolution. No finite differences or interpolation in time.
(function () {
    'use strict';
    function samples(value) {
        if(Array.isArray(value))return value;
        if(!value || !value.bdata)return [];
        const bytes=Uint8Array.from(atob(value.bdata),c=>c.charCodeAt(0));
        const type={f8:Float64Array,f4:Float32Array,i4:Int32Array,u4:Uint32Array,i2:Int16Array,u2:Uint16Array,i1:Int8Array,u1:Uint8Array}[value.dtype];
        return type?new type(bytes.buffer):[];
    }
    function fresnel(x,table) {
        const sign=x<0?-1:1,u=Math.abs(x);
        if(u<=6) {
            const h=table.step,i=Math.min(table.c.length-2,Math.floor(u/h)),v=(u-i*h)/h;
            const h00=2*v*v*v-3*v*v+1,h10=v*v*v-2*v*v+v,h01=-2*v*v*v+3*v*v,h11=v*v*v-v*v;
            const p0=Math.PI*(i*h)**2/2,p1=Math.PI*((i+1)*h)**2/2;
            return [sign*(h00*table.c[i]+h10*h*Math.cos(p0)+h01*table.c[i+1]+h11*h*Math.cos(p1)),sign*(h00*table.s[i]+h10*h*Math.sin(p0)+h01*table.s[i+1]+h11*h*Math.sin(p1))];
        }
        const den=Math.PI*Math.PI*u**4;let f=1,g=1,tf=1,tg=1;
        for(let n=1;n<=12;n++){tf*=-(4*n-1)*(4*n-3)/den;tg*=-(4*n+1)*(4*n-1)/den;f+=tf;g+=tg;}
        f/=Math.PI*u;g/=Math.PI*Math.PI*u**3;
        const p=Math.PI*u*u/2;
        return [sign*(.5+f*Math.sin(p)-g*Math.cos(p)),sign*(.5-f*Math.cos(p)-g*Math.sin(p))];
    }
    function frame(m,t) {
        let x=m.x, re=[], im=[], mean=0, sd=0, sdP=0, norm=1;
        const offset=m.mode==='oscillator'?m.energy:0, amp=m.amp || 1;
        if (m.mode==='plane') {
            for(const v of x) {const p=m.k*v-m.k*m.k*t/(2*m.mass);re.push(Math.cos(p));im.push(Math.sin(p));}
        } else if(m.mode==='free') {
            const sx=1/(2*m.sk), b=t/(2*m.mass*sx*sx),den=1+b*b;
            const amplitude=Math.pow(den,-.25),spatial=1/(4*sx*sx*den),phaseOffset=-.5*Math.atan(b)-m.k*m.k*t/(2*m.mass);
            mean=m.k*t/m.mass;sd=sx*Math.sqrt(1+b*b);sdP=m.sk;
            if(m.options.includes('follow')) {
                const span=Math.max(15,6*sd);x=Array.from({length:1801},(_,i)=>mean-span+2*span*i/1800);
            }
            for(const v of x) {
                const d=v-mean, a=amplitude*Math.exp(-d*d*spatial);
                const p=phaseOffset+d*d*b*spatial+m.k*v;
                re.push(a*Math.cos(p));im.push(a*Math.sin(p));
            }
        } else if(m.mode==='rect') {
            mean=m.k*t/m.mass;sd=null;sdP=null;
            for(const v of x) {
                let r=0,i=0;
                if(t===0) {r=Math.abs(v)<m.a?1:0;}
                else {
                    const scale=Math.sqrt(m.mass/(Math.PI*t));
                    const fp=fresnel((v-mean+m.a)*scale,m.fresnel),fm=fresnel((v-mean-m.a)*scale,m.fresnel);
                    const dc=fp[0]-fm[0],ds=fp[1]-fm[1];r=(dc+ds)/2;i=(ds-dc)/2;
                }
                const p=m.k*v-m.k*m.k*t/(2*m.mass);
                re.push(r*Math.cos(p)-i*Math.sin(p));im.push(r*Math.sin(p)+i*Math.cos(p));
            }
        } else {
            const cr=m.cr.map((v,n)=>v*Math.cos(m.energies[n]*t)+m.ci[n]*Math.sin(m.energies[n]*t));
            const ci=m.ci.map((v,n)=>v*Math.cos(m.energies[n]*t)-m.cr[n]*Math.sin(m.energies[n]*t));
            const active=cr.map((v,n)=>n).filter(n=>cr[n]!==0 || ci[n]!==0);
            for(let j=0;j<x.length;j++) {let r=0,i=0;for(const n of active){r+=cr[n]*m.basis[n][j];i+=ci[n]*m.basis[n][j];}re.push(r/amp);im.push(i/amp);}
            if(m.mode==='oscillator') {
                let number=0, ar=0, ai=0, a2r=0;
                for(let n=0;n<cr.length;n++) {
                    number+=n*(cr[n]*cr[n]+ci[n]*ci[n]);
                    if(n>0){ar+=Math.sqrt(n)*(cr[n-1]*cr[n]+ci[n-1]*ci[n]);ai+=Math.sqrt(n)*(cr[n-1]*ci[n]-ci[n-1]*cr[n]);}
                    if(n>1)a2r+=Math.sqrt(n*(n-1))*(cr[n-2]*cr[n]+ci[n-2]*ci[n]);
                }
                mean=Math.sqrt(2)*ar;sd=Math.sqrt(Math.max(0,number+.5+a2r-mean*mean));
                sdP=Math.sqrt(Math.max(0,number+.5-a2r-2*ai*ai));
            } else {
                let second=0;norm=0;
                for(let j=1;j<x.length;j++) {
                    const r0=re[j-1]**2+im[j-1]**2,r1=re[j]**2+im[j]**2,dx=(x[j]-x[j-1])/2;
                    norm+=dx*(r0+r1);mean+=dx*(x[j-1]*r0+x[j]*r1);second+=dx*(x[j-1]**2*r0+x[j]**2*r1);
                }
                sd=Math.sqrt(Math.max(0,second-mean*mean));
            }
        }
        return {x,re,im,density:re.map((r,i)=>r*r+im[i]*im[i]),phase:m.options.includes('phase')?re.map((r,i)=>Math.atan2(im[i],r)):[],mean,sd,sdP,norm,offset};
    }
    function metric(label,value) {return {namespace:'dash_html_components',type:'Div',props:{className:'metric',children:[{namespace:'dash_html_components',type:'Small',props:{children:label}},{namespace:'dash_html_components',type:'Strong',props:{children:value}}]}};}
    function dispersionFrame(m,t) {
        const sx=1/(2*m.sk),x=m.x;
        const matter=frame(Object.assign({},m,{mode:'free'}),t);
        const re=[],im=[],density=[];
        for(const v of x) {
            const d=v-m.k*t/m.mass,a=Math.exp(-d*d/(4*sx*sx)),p=m.k*d;
            re.push(a*Math.cos(p));im.push(a*Math.sin(p));density.push(a*a);
        }
        return [{x,re,im,density,mean:m.k*t/m.mass,sd:sx},matter];
    }
    function render(t,m,base,mode) {
        const no=window.dash_clientside.no_update;
        if(!m || m.mode!==mode || !base || !base.data || !base.data.length)return [no,no,no];
        if(mode==='dispersion') {
            if(base.layout.uirevision!=='dispersion')return [no,no,no];
            const frames=dispersionFrame(m,Number(t||0)),sx=1/(2*m.sk),x=m.x;
            const data=base.data.map(tr=>{
                const meta=tr.meta||{},f=frames[meta.row-1];if(!f)return tr;
                let y;
                if(meta.field==='real')y=f.re;
                else if(meta.field==='density')y=f.density;
                else if(meta.field==='reference')y=x.map(v=>Math.exp(-((v-f.mean)**2)/(2*sx*sx)));
                else y=f.density.map(v=>Math.sqrt(v)*(meta.field==='neg-env'?-1:1));
                return Object.assign({},tr,{x,y});
            });
            const layout=JSON.parse(JSON.stringify(base.layout));
            frames.forEach((f,i)=>{
                const center=(layout.shapes||[])[2*i],band=(layout.shapes||[])[2*i+1];
                if(center)center.x0=center.x1=f.mean;
                if(band){band.x0=f.mean-f.sd;band.x1=f.mean+f.sd;}
                const a=(layout.annotations||[])[2+i];
                if(a){a.x=f.mean;a.text='폭 σ<sub>x</sub> = '+f.sd.toFixed(3)+' L₀ · 초기의 '+(f.sd/sx).toFixed(2)+'배';}
            });
            const metrics=[metric('무분산 · 폭 / 초기 폭','1.000'),metric('물질파 · 폭 / 초기 폭',(frames[1].sd/sx).toFixed(3)),metric('무분산 · 중심 속도',(m.k/m.mass).toFixed(3)+' V₀'),metric('물질파 · 중심 속도',(m.k/m.mass).toFixed(3)+' V₀')];
            return [{data,layout},metrics,Number(t||0).toFixed(2)];
        }
        if(mode==='beat') {
            if(base.layout.uirevision!=='beat')return [no,no,no];
            const k=m.k,dk=m.sk,k2=k+dk,w1=m.kind==='linear'?k:k*k/(2*m.mass),w2=m.kind==='linear'?k2:k2*k2/(2*m.mass);
            const vg=(w2-w1)/dk,vp=(w1+w2)/(k+k2),x=m.x;
            const ys=[x.map(v=>Math.cos(k*v-w1*t)),x.map(v=>Math.cos(k2*v-w2*t))];
            ys.push(ys[0].map((v,i)=>v+ys[1][i]));
            const env=x.map(v=>2*Math.abs(Math.cos((dk*v-(w2-w1)*t)/2)));
            if(m.options.includes('envelope'))ys.push(env,env.map(v=>-v));
            const speeds=[vg,vp,w1/k,w2/k2];
            const data=base.data.map((tr,i)=>{
                if(tr.name==='마루 추적 1' || tr.name==='마루 추적 2') {
                    const speed=tr.name==='마루 추적 1'?speeds[2]:speeds[3];
                    return Object.assign({},tr,{x:[speed*t],y:[1]});
                }
                return Object.assign({},tr,{x,y:ys[i]});
            });
            const layout=JSON.parse(JSON.stringify(base.layout));
            (layout.shapes||[]).forEach((v,i)=>v.x0=v.x1=speeds[i]*t);
            const a=layout.annotations||[];
            for(let i=0;i<4;i++)if(a[i+2])a[i+2].x=speeds[i]*t;
            return [{data,layout},no,Number(t||0).toFixed(2)];
        }
        const expected={plane:'plane',free:'free',well:'well',oscillator:'oscillator',rect:'rect'}[mode];
        if(base.layout.uirevision!==expected)return [no,no,no];
        const f=frame(m,Number(t||0)), layout=JSON.parse(JSON.stringify(base.layout));
        const data=base.data.map(tr=>{
            const out=Object.assign({},tr),name=tr.name||'';
            if(name.includes('퍼텐셜'))return out;
            if(name.includes('t=0')) {
                if(mode==='free' && m.options.includes('follow')) {out.x=f.x;out.y=f.x.map(v=>Math.exp(-v*v*2*m.sk*m.sk));}
                return out;
            }
            let y;
            if(name.includes('색 = 위상')) {
                const stride=Math.max(1,Math.ceil(f.x.length/480));
                const indices=Array.from({length:Math.ceil(f.x.length/stride)},(_,i)=>i*stride);
                out.x=indices.map(i=>f.x[i]);
                y=indices.map(i=>mode==='plane'?f.re[i]:Math.hypot(f.re[i],f.im[i])>.01?f.offset+Math.hypot(f.re[i],f.im[i]):null);
                out.marker=Object.assign({},tr.marker,{color:indices.map(i=>f.phase[i])});
                out.customdata=indices.map(i=>mode==='oscillator'?[Math.hypot(f.re[i],f.im[i]),f.phase[i]]:f.phase[i]);
            } else {
                const values=name.includes('Re[')?f.re:name.includes('Im[')?f.im:name.includes('포락선')?f.density.map(v=>Math.sqrt(v)*(name.includes('음의')?-1:1)):f.density;
                out.x=f.x; y=values.map(v=>v+f.offset);
                if(mode==='oscillator') {
                    out.customdata=values;
                    if(name.includes('|Ψ|²')) {out.x=f.x.concat(f.x.slice().reverse());y=y.concat(f.x.map(()=>f.offset));out.customdata=values.concat(f.x.map(()=>0));}
                }
            }
            out.y=y;return out;
        });
        const shapes=layout.shapes||[],annotations=layout.annotations||[];
        if(mode!=='plane') {
            const index=mode==='well'?2:mode==='oscillator'?1:0;
            if(shapes[index])shapes[index].x0=shapes[index].x1=f.mean;
            if(annotations[0])annotations[0].x=f.mean;
            if(mode==='rect' && annotations[0]){annotations[0].xanchor=f.mean>0?'right':'left';annotations[0].xshift=f.mean>0?-8:8;}
            if(mode==='free') {
                if(shapes[1]){shapes[1].x0=f.mean-f.sd;shapes[1].x1=f.mean+f.sd;}
                layout.xaxis.range=[f.x[0],f.x[f.x.length-1]];
                const left=f.mean>f.x[0]+.82*(f.x[f.x.length-1]-f.x[0]);
                if(annotations[0]){annotations[0].xanchor=left?'right':'left';annotations[0].xshift=left?-8:8;}
            }
        }
        if(m.axis==='auto') {
            let lo=0,hi=0;
            for(const tr of data)if(!String(tr.name||'').includes('퍼텐셜') && tr.visible!==false && tr.visible!=='legendonly')for(const y of samples(tr.y))if(Number.isFinite(y)){lo=Math.min(lo,y-f.offset);hi=Math.max(hi,y-f.offset);}
            if(mode==='oscillator') {
                const a=Math.min(-.25,f.offset+lo-.15),b=Math.max(4,f.offset+hi+Math.max(.3,.1*hi));
                layout.yaxis.range=[f.offset+(a-f.offset)/m.zoom,f.offset+(b-f.offset)/m.zoom];
            } else if(mode==='well')layout.yaxis.range=[(lo-.15)/m.zoom,(Math.max(.5,hi)+.2)/m.zoom];
            else {const edge=Math.max(Math.abs(lo),Math.abs(hi),1e-8)*1.12/m.zoom;layout.yaxis.range=lo< -1e-9?[-edge,edge]:[-.035*edge,edge];if(mode==='plane')layout.yaxis.range[1]=Math.max(1.95,layout.yaxis.range[1]);}
        }
        let metrics=no;
        if(mode==='rect')metrics=[metric('대칭 중심 x_c',f.mean.toFixed(3)),metric('중심 이동 속도 v_c / v₀',(m.k/m.mass).toFixed(3)),metric('초기 위치 표준편차 Δx₀',(m.a/Math.sqrt(3)).toFixed(3)),metric('파수 중심 봉우리 폭 · 첫 영점 사이',(2*Math.PI/m.a).toFixed(3))];
        if(mode==='free')metrics=[metric('평균 위치 ⟨x⟩',f.mean.toFixed(3)),metric('위치 표준편차 Δx',f.sd.toFixed(3)),metric('파수 표준편차 Δk',m.sk.toFixed(3)),metric('현재 Δx(t)·Δp / ℏ',(f.sd*m.sk).toFixed(5))];
        if(mode==='well')metrics=[metric('평균 위치 ⟨x⟩ / L',f.mean.toFixed(3)),metric('위치 표준편차 Δx / L',f.sd.toFixed(3)),metric('평균 에너지 / E₁',m.energy.toFixed(3)),metric('확률 총합',f.norm.toFixed(5))];
        if(mode==='oscillator')metrics=[metric('평균 위치 ⟨x⟩',f.mean.toFixed(3)),metric('위치 표준편차 Δx',f.sd.toFixed(3)),metric('현재 Δx·Δp / ℏ',(f.sd*f.sdP).toFixed(5)),metric('평균 에너지 / ℏω',m.energy.toFixed(3))];
        return [{data,layout},metrics,Number(t||0).toFixed(2)];
    }
    window.quantumPlayback={frame,render,fresnel,dispersionFrame};
    window.dash_clientside=Object.assign({},window.dash_clientside,{quantum:{render}});
})();
