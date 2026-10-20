"""Two comparison figures, before vs after calibration.
 A: scaffold molecules at the dominant pole  vs  the localisation ratio achieved
 B: the localisation ratio  vs  the coverage theta
Data from s1s2_curves.py (run that first).
"""
import os as _os, sys as _sys, json
_HERE=_os.path.dirname(_os.path.abspath(__file__)); ROOT=_os.path.dirname(_HERE)
FIGS=_os.path.join(ROOT,'figures')
_os.makedirs(FIGS, exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=json.load(open(_os.path.join(FIGS,'_s1s2_curves.json')))
SURF,INK,INK2,MUTED,GRID='#fcfcfb','#0b0b0b','#52514e','#8a8882','#e4e2dd'
S1,S2N,S2M='#2a78d6','#eb6834','#1baf7a'     # validated 3-hue set (light mode)
PRE='before calibration   L 3.0 µm, p 0.5, ρ$_n$ 2.370'
POST='after calibration   L 2.78 µm, p 0.17, ρ$_n$ 4.181'
def frame(a):
    a.set_facecolor(SURF)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
    for sp in ('left','bottom'): a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2,labelsize=9)

# ===================== A: scaffold vs ratio =====================
fig,ax=plt.subplots(figsize=(9.0,5.1),dpi=200); fig.patch.set_facecolor(SURF); frame(ax)
for g in (25,50,75,100,125,150): ax.axhline(g,color=GRID,lw=1,zorder=0)
for key,ls,wid in (('pre',':',2.0),('post','-',2.2)):
    e=D[key]
    ax.axhline(e['baseline_scaffold'],color=MUTED,lw=1.1,ls=ls,zorder=1)
    for tag,c in (('s1',S1),('s2mn',S2N),('s2m',S2M)):
        lr=np.array(e[tag]['lr']); sc=np.array(e[tag]['sc']); m=lr>=1.0
        ax.plot(lr[m],sc[m],ls,color=c,lw=wid,zorder=4,solid_capstyle='round')
ax.text(31,D['post']['baseline_scaffold']+4.5,'no PopZ:  109 / 113 molecules per pole',
        fontsize=8.8,color=MUTED,ha='right')
ax.text(13,152,'S1   steric occupancy',fontsize=11.5,color=S1,ha='center',fontweight='bold')
ax.text(13,52,'S2   local degradation\n(clusters cleavable too)',fontsize=11.5,color=S2N,
        ha='center',linespacing=1.5,fontweight='bold')
ax.text(1.42,10,'S2   local degradation (monomer only)\nno scaffold survives  (< 0.2 molecules)',
        fontsize=9.6,color=S2M,ha='left',linespacing=1.5)
ax.set_xscale('log'); ax.set_xlim(1.3,32); ax.set_ylim(0,168)
ax.set_xticks([2,3,5,10,20,30]); ax.set_xticklabels(['2','3','5','10','20','30'])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('localisation ratio achieved   (cluster molecules: good pole / blocked pole)',
              fontsize=10.5,color=INK2)
ax.set_ylabel('scaffold left at the good pole   [molecules]',fontsize=10.5,color=INK2)
ax.set_title('For a given localisation ratio, how much scaffold is left?',
             fontsize=13,color=INK,loc='left',pad=12)
h=[plt.Line2D([],[],color=INK2,ls=':',lw=2.0,label=PRE),
   plt.Line2D([],[],color=INK2,ls='-',lw=2.2,label=POST)]
ax.legend(handles=h,frameon=False,fontsize=9,labelcolor=INK2,loc='center left',bbox_to_anchor=(.02,.57))
fig.text(.012,-.035,
 'Steady state with PopZ held on. The test is not the left/right ratio but whether a scaffold survives at one pole — so higher is better.'
 '\n⚠️ After calibration S1 cannot reach a ratio of 10 even at θ → 1 (ceiling 9.9×; it was 27.5× before).'
 '\n★ The S1 curves nearly coincide: once a ratio is reached, the scaffold is the same. Calibration changed the cost, not the payoff.',
 fontsize=8.4,color=MUTED,ha='left',linespacing=1.6)
fig.tight_layout(rect=[0,.06,1,1])
fig.savefig(_os.path.join(FIGS,'cal_fig_s1s2_scaffold.png'),facecolor=SURF,bbox_inches='tight')

# ===================== B: ratio vs theta =====================
fig,ax=plt.subplots(figsize=(9.0,5.1),dpi=200); fig.patch.set_facecolor(SURF); frame(ax)
th=np.linspace(0,.985,300)
ax.plot(th,1/(1-th),'--',color=MUTED,lw=1.5,zorder=2)
ax.text(.45,8.6,'binding-capacity ratio  1/(1−θ)\n(molecules splitting in simple proportion to capacity)',
        fontsize=9,color=MUTED,ha='center',linespacing=1.5)
for key,ls,wid in (('pre',':',2.1),('post','-',2.3)):
    e=D[key]['s1']; ax.plot(e['theta'],e['lr'],ls,color=S1,lw=wid,zorder=5,solid_capstyle='round')
for tgt in (2,3,5):
    ax.axhline(tgt,color=GRID,lw=1,zorder=0)
    ax.text(.012,tgt*1.04,'%d×'%tgt,fontsize=8.8,color=MUTED,va='bottom')
for x,y,t,c,dy in ((.350,2,'0.35',MUTED,-17),(.504,3,'0.50',MUTED,-17),(.659,5,'0.66',MUTED,-17),
                   (.517,2,'0.52',INK,-17),(.702,3,'0.70',INK,-17),(.861,5,'0.86',INK,12)):
    ax.plot([x],[y],'o',ms=6.5,mfc=SURF,mec=S1,mew=1.8,zorder=6)
    ax.annotate(t,(x,y),textcoords='offset points',xytext=(0,dy),ha='center',fontsize=8.8,color=c,zorder=7)
ax.text(.70,11.5,'before',fontsize=11.5,color=S1,ha='center',alpha=.6,fontweight='bold')
ax.text(.945,3.3,'after',fontsize=11.5,color=S1,ha='center',fontweight='bold')
ax.set_yscale('log'); ax.set_xlim(0,1.0); ax.set_ylim(.95,40)
ax.set_yticks([1,2,3,5,10,20,30]); ax.set_yticklabels(['1','2','3','5','10','20','30'])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('effective coverage  θ    (fraction of binding capacity removed at the blocked pole)',
              fontsize=10.5,color=INK2)
ax.set_ylabel('localisation ratio   (good pole / blocked pole)',fontsize=10.5,color=INK2)
ax.set_title('Calibration raised the coverage needed for the same ratio by ~1.4×',
             fontsize=13,color=INK,loc='left',pad=12)
h=[plt.Line2D([],[],color=S1,ls=':',lw=2.1,label=PRE),
   plt.Line2D([],[],color=S1,ls='-',lw=2.3,label=POST)]
ax.legend(handles=h,frameon=False,fontsize=9,labelcolor=INK2,loc='upper left')
fig.text(.012,-.035,
 '★ Before calibration the curve ran ABOVE the dashed line (up to 1.83× at θ 0.8): molecules were more skewed than the binding capacity alone, '
 'because clusters collapsing at the blocked pole feed growth at the other one.'
 '\n⚠️ After calibration that self-amplification is essentially gone. The curve tracks the dashed line up to θ ≈ 0.3 and falls below it beyond '
 '(0.80× at θ 0.8, 0.59× at θ 0.9), and the θ → 1 ceiling drops from 27.5× to 9.9×.'
 '\n   The cause is p falling from 0.5 to 0.17 — with little cooperativity left, blocking has to do the work on its own.',
 fontsize=8.4,color=MUTED,ha='left',linespacing=1.6)
fig.tight_layout(rect=[0,.06,1,1])
fig.savefig(_os.path.join(FIGS,'cal_fig_s1_theta.png'),facecolor=SURF,bbox_inches='tight')
print('wrote both')
