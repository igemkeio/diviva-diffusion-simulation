import os as _os
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.makedirs(_os.path.join(_ROOT, 'figures'), exist_ok=True)
"""How much of the predicted localisation actually shows up in the data,
   before and after calibration.  Excess = end_mid - baseline."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=_os.path.join(_ROOT,'figures','cal_fig_agreement.png')
SURF,INK,INK2,MUTED,GRID='#fcfcfb','#0b0b0b','#52514e','#8a8882','#e4e2dd'
MODEL,MEAS='#eb6834','#2a78d6'          # validated pair (6/6 checks, light mode)

ROWS=[  # label, baseline, model end_mid, measured end_mid
 ('before calibration\nL 3.0 µm · p 0.5 · ρn 2.370', 0.6713, 0.8427, 0.7539),
 ('after calibration\nL 2.78 µm · p 0.17 · ρn 4.181', 0.6399, 0.7977, 0.7539),
]
fig,ax=plt.subplots(figsize=(9.4,3.9),dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
H=0.30; GAP=0.025                      # bar height, 2px-equivalent surface gap
def bar(x0,y,w,h,c):
    ax.add_patch(FancyBboxPatch((x0,y-h/2),max(w,1e-4),h,
        boxstyle='round,pad=0,rounding_size=0.004',mutation_aspect=0.06,
        fc=c,ec='none',zorder=3,clip_on=False))
ys=[]
for i,(lab,b,mo,me) in enumerate(ROWS):
    y=-i*1.0; ys.append(y)
    em,ee=mo-b,me-b
    bar(0,y+H/2+GAP/2,em,H,MODEL)
    bar(0,y-H/2-GAP/2,ee,H,MEAS)
    ax.text(em+0.004,y+H/2+GAP/2,'%.3f'%em,va='center',ha='left',fontsize=9.5,color=INK2)
    ax.text(ee+0.004,y-H/2-GAP/2,'%.3f'%ee,va='center',ha='left',fontsize=9.5,color=INK2)
    ax.text(-0.006,y,lab,va='center',ha='right',fontsize=9.5,color=INK,linespacing=1.6)
    ax.text(0.218,y,'%.0f%%'%(100*ee/em),va='center',ha='right',
            fontsize=24,color=INK if i else MUTED)
ax.text(0.218,ys[0]+0.47,'of the model’s prediction\nis in the data',
        va='bottom',ha='right',fontsize=8.6,color=MUTED,linespacing=1.5)
ax.annotate('',xy=(0.1985,ys[1]+0.30),xytext=(0.1985,ys[0]-0.30),
            arrowprops=dict(arrowstyle='-|>',color=GRID,lw=2.0,shrinkA=2,shrinkB=2))
ax.set_ylim(ys[1]-0.62,ys[0]+0.80)
ax.set_yticks([])
for sp in ('top','right','left'): ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(MUTED)
ax.set_xticks(np.arange(0,0.19,0.05))
ax.set_xlim(0,0.238)
ax.tick_params(axis='x',colors=INK2,labelsize=9)
ax.set_xlabel('localisation excess   (end/mid − the no-localisation baseline)',fontsize=10,color=INK2)
for g in np.arange(0.05,0.19,0.05): ax.axvline(g,color=GRID,lw=1,zorder=0)
ax.axvline(0,color=MUTED,lw=1.2,zorder=2)
h1=plt.Line2D([],[],marker='s',ms=9,ls='none',color=MODEL,label='model predicts')
h2=plt.Line2D([],[],marker='s',ms=9,ls='none',color=MEAS,label='measured  (DivIVA-mCherry, n = 460)')
ax.legend(handles=[h1,h2],frameon=False,fontsize=9,labelcolor=INK2,
          loc='lower left',bbox_to_anchor=(0.0,1.01),ncol=2,handletextpad=.5,columnspacing=1.8)
fig.text(.012,-.07,
 'Line width 5 px, tip window z 0–0.144, effective PSF 0.870 µm. The baseline (end/mid with no localisation anywhere, membrane 70%) '
 'is 0.6713 before and 0.6399 after, because it depends on cell length.\nThe model’s own prediction barely moved (0.171 → 0.158); what '
 'changed is the baseline, and with it how much of the measurement counts as localisation. Before-calibration is a reconstruction — that model had no forward model.',
 fontsize=8,color=MUTED,ha='left',linespacing=1.6)
fig.tight_layout(rect=[0,.06,1,.99])
fig.savefig(OUT,facecolor=SURF,bbox_inches='tight')
print('wrote',OUT)
for lab,b,mo,me in ROWS: print('  %-22s 超過 model %.4f  meas %.4f  -> %.0f%%'%(lab.split('\n')[0],mo-b,me-b,100*(me-b)/(mo-b)))
