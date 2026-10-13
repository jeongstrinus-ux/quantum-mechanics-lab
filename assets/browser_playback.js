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
    function frame(m,t) {
        let x=m.x, re=[], im=[], mean=0, sd=0, sdP=0, norm=1;
        const offset=m.mode==='oscillator'?m.energy:0, amp=m.amp || 1;
        if (m.mode==='plane') {
            for(const v of x) {const p=m.k*v-m.k*m.k*t/(2*m.mass);re.push(Math.cos(p));im.push(Math.sin(p));}
        } else if(m.mode==='free') {
            const sx=1/(2*m.sk), b=t/(2*m.mass*sx*sx);
            mean=m.k*t/m.mass;sd=sx*Math.sqrt(1+b*b);sdP=m.sk;
            if(m.options.includes('follow')) {
                const span=Math.max(15,6*sd);x=Array.from({length:1801},(_,i)=>mean-span+2*span*i/1800);
            }
            for(const v of x) {
                const d=v-mean, a=Math.pow(1+b*b,-.25)*Math.exp(-d*d/(4*sx*sx*(1+b*b)));
                const p=-.5*Math.atan(b)+d*d*b/(4*sx*sx*(1+b*b))+m.k*v-m.k*m.k*t/(2*m.mass);
                re.push(a*Math.cos(p));im.push(a*Math.sin(p));
            }
        } else {
            const cr=m.cr.map((v,n)=>v*Math.cos(m.energies[n]*t)+m.ci[n]*Math.sin(m.energies[n]*t));
            const ci=m.ci.map((v,n)=>v*Math.cos(m.energies[n]*t)-m.cr[n]*Math.sin(m.energies[n]*t));
            for(let j=0;j<x.length;j++) {let r=0,i=0;for(let n=0;n<cr.length;n++){r+=cr[n]*m.basis[n][j];i+=ci[n]*m.basis[n][j];}re.push(r/amp);im.push(i/amp);}
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
        return {x,re,im,density:re.map((r,i)=>r*r+im[i]*im[i]),phase:re.map((r,i)=>Math.atan2(im[i],r)),mean,sd,sdP,norm,offset};
    }
    function metric(label,value) {return {namespace:'dash_html_components',type:'Div',props:{className:'metric',children:[{namespace:'dash_html_components',type:'Small',props:{children:label}},{namespace:'dash_html_components',type:'Strong',props:{children:value}}]}};}
    function render(t,m,base,mode) {
        const no=window.dash_clientside.no_update;
        if(!m || m.mode!==mode || !base || !base.data || !base.data.length)return [no,no,no];
        const expected={plane:'plane',free:'free',well:'well',oscillator:'oscillator'}[mode];
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
        if(mode==='free')metrics=[metric('평균 위치 ⟨x⟩',f.mean.toFixed(3)),metric('위치 표준편차 Δx',f.sd.toFixed(3)),metric('파수 표준편차 Δk',m.sk.toFixed(3)),metric('현재 Δx(t)·Δp / ℏ',(f.sd*m.sk).toFixed(5))];
        if(mode==='well')metrics=[metric('평균 위치 ⟨x⟩ / L',f.mean.toFixed(3)),metric('위치 표준편차 Δx / L',f.sd.toFixed(3)),metric('평균 에너지 / E₁',m.energy.toFixed(3)),metric('확률 총합',f.norm.toFixed(5))];
        if(mode==='oscillator')metrics=[metric('평균 위치 ⟨x⟩',f.mean.toFixed(3)),metric('위치 표준편차 Δx',f.sd.toFixed(3)),metric('현재 Δx·Δp / ℏ',(f.sd*f.sdP).toFixed(5)),metric('평균 에너지 / ℏω',m.energy.toFixed(3))];
        return [{data,layout},metrics,Number(t||0).toFixed(2)];
    }
    window.quantumPlayback={frame,render};
    window.dash_clientside=Object.assign({},window.dash_clientside,{quantum:{render}});
})();
