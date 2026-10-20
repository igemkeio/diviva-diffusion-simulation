"""Data for the two comparison figures: scaffold vs ratio, and ratio vs theta.
   Writes figures/_s1s2_curves.json so the plotting scripts need no solver."""
import os as _os, sys as _sys, json
_HERE=_os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path: _sys.path.insert(0,_HERE)
import numpy as np, popz
_FIGS=_os.path.join(_os.path.dirname(_HERE),'figures')
_os.makedirs(_FIGS, exist_ok=True)
OUT=_os.path.join(_FIGS,'_s1s2_curves.json')
CASES={'pre' :dict(base=dict(R=0.40,L=3.0,   p=0.50,rho_n=2.3696),n=210),
       'post':dict(base=dict(R=0.40,L=2.7767,p=0.17,rho_n=4.1812),n=236)}
TH =[0,.1,.2,.3,.4,.5,.55,.6,.65,.7,.75,.8,.85,.9,.93,.96,.98,.99,.995]
KC_MN=list(np.logspace(-4,-1.3,14)); KC_M=list(np.logspace(-1,1.2,12))
d={}
for k,cf in CASES.items():
    b,n=cf['base'],cf['n']
    base=popz.steady(n,popz.variant(popz_mode='cap',theta=0.0,k_cut=0.0,**b))
    e={'baseline_scaffold':base['scaffold_R'],'enr_clu':base['enr_clu'],
       's1':{'theta':[],'lr':[],'sc':[]},
       's2mn':{'kcut':[],'lr':[],'sc':[],'mass':[]},
       's2m':{'kcut':[],'lr':[],'sc':[],'mass':[]}}
    for th in TH:
        r=popz.steady(n,popz.variant(popz_mode='cap',theta=th,k_cut=0.0,**b))
        e['s1']['theta'].append(th); e['s1']['lr'].append(r['lr_clu']); e['s1']['sc'].append(r['scaffold_R'])
    for tag,kcs,tgt in (('s2mn',KC_MN,'MN'),('s2m',KC_M,'M')):
        for kc in kcs:
            r=popz.steady(n,popz.variant(popz_mode='cap',theta=0.0,k_cut=kc,cut_target=tgt,**b))
            e[tag]['kcut'].append(float(kc)); e[tag]['lr'].append(r['lr_clu'])
            e[tag]['sc'].append(r['scaffold_R']); e[tag]['mass'].append(r['mass_frac'])
    d[k]=e
    print(k,'done  baseline scaffold %.1f'%base['scaffold_R'])
json.dump(d,open(OUT,'w'),indent=1)
print('wrote',OUT)
