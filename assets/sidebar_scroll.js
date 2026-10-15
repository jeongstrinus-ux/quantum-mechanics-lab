// Fit the sidebar into the currently visible viewport, including before sticky
// positioning reaches the top. No simulation state or animation timer changes.
(function () {
    'use strict';
    let panel=null,pending=false;
    function update() {
        pending=false;
        if(!panel || !panel.isConnected)return;
        if(window.innerWidth<=950){panel.style.removeProperty('--controls-available-height');return;}
        const top=Math.max(12,panel.getBoundingClientRect().top);
        const height=Math.max(120,window.innerHeight-top-12)+'px';
        if(panel.style.getPropertyValue('--controls-available-height')!==height)
            panel.style.setProperty('--controls-available-height',height);
    }
    function schedule() {
        if(!pending){pending=true;window.requestAnimationFrame(update);}
    }
    const ready=window.setInterval(function () {
        panel=document.querySelector('main>.controls');
        if(!panel)return;
        window.clearInterval(ready);
        window.addEventListener('scroll',schedule,{passive:true});
        window.addEventListener('resize',schedule,{passive:true});
        if(window.ResizeObserver) {
            const observer=new ResizeObserver(schedule);
            for(const selector of ['header','.experiment-nav','.intro','main']) {
                const element=document.querySelector(selector);if(element)observer.observe(element);
            }
        }
        schedule();
    },100);
})();
