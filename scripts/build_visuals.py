"""Render measured results and the real offline demo report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from netops_fault_lab import Case, diagnose
from netops_fault_lab.report import render_html
from netops_fault_lab.schema import read_json

root=Path(__file__).resolve().parents[1]
assets=root/'docs/assets'
assets.mkdir(exist_ok=True)
result=json.loads((root/'benchmarks/results/v0.1/results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':'#284652','text.color':'#193b47',
                     'axes.edgecolor':'#c9d6dc','xtick.color':'#49626d','ytick.color':'#49626d'})
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
fig.patch.set_facecolor('#f6f9fa')
for ax in axes:
 ax.set_facecolor('#f6f9fa'); ax.grid(axis='y',color='#e0e8eb',linewidth=.7); ax.set_axisbelow(True)
methods=[('failed_path_votes','Failed-path votes','#b5c5cc'),('logistic','Logistic regression','#c38c3d'),('bayes','Bayesian','#147c76')]
scenarios=['nominal','noise_shift','missing_55pct','copied_alarms_x4']
x=np.arange(4)
for i,(method,label,color) in enumerate(methods):
 values=[100*result['summary'][s][method]['top1'] for s in scenarios]
 axes[0].bar(x+(i-1)*.23,values,width=.21,label=label,color=color)
axes[0].set_xticks(x,['Nominal','Noise shift','55% missing','Copied alarms'])
axes[0].set_ylim(0,100); axes[0].set_ylabel('Top-1 accuracy (%)')
axes[0].set_title('Diagnosis under four conditions',loc='left',pad=18,fontweight='bold')
axes[0].legend(frameon=False,fontsize=8,loc='upper right')
for policy,label,color in [('information_gain','Expected information gain','#147c76'),('random','Random next probe','#c38c3d')]:
 values=[100*result['active_probes'][policy][str(i)]['top1'] for i in range(5)]
 axes[1].plot(range(5),values,'o-',label=label,color=color,linewidth=2.3,markersize=5)
 axes[1].annotate(f'{values[-1]:.1f}%',(4,values[-1]),xytext=(6,0),textcoords='offset points',color=color,va='center',fontweight='bold')
axes[1].set_ylim(20,90); axes[1].set_xlim(-.1,4.65); axes[1].set_xticks(range(5))
axes[1].set_ylabel('Top-1 accuracy (%)');axes[1].set_xlabel('Additional probes (same candidate pool and sampled outcomes)')
axes[1].set_title('Which path should we measure next?',loc='left',pad=18,fontweight='bold')
axes[1].legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('Synthetic experiments · 3 seeds · 864 held-out incidents per condition',fontsize=13,fontweight='bold')
fig.savefig(assets/'benchmark.png',dpi=180)
plt.close(fig)
for filename in ['ambiguous','after-probe']:
 case=Case.from_dict(read_json(root/f'examples/{filename}.json'))
 report=diagnose(case)
 (root/f'docs/{filename}.html').write_text(render_html(case,report),encoding='utf-8')
 (root/f'examples/{filename}-result.json').write_text(json.dumps(report,indent=2)+'\n')
print('Rendered benchmark.png, two HTML reports and their JSON results.')
