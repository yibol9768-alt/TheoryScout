"""Derived figures only; frozen forecasts and final seed intervals, no fitting."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from P1NUM_run import ROOT,OUT,moments,F,put,metadata,frozen

def read(name):return list(csv.DictReader((OUT/name).open()))
def main():
 frozen();meta=metadata();a=read('P1NUM_A_gates.csv');m=read('P1NUM_M_gates.csv');fc=json.loads((OUT/'P1NUM_forecasts.json').read_text())
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
 for ax,g in zip(axs,(16,64,256)):
  for d,color,mark in ((128,'#348ABD','o'),(256,'#E24A33','s')):
   rr=sorted([r for r in a if int(r['gamma'])==g and int(r['d'])==d and r['label']!='physical_epsilon'],key=lambda r:float(r['e']))
   ax.errorbar([float(r['e']) for r in rr],[100*float(r['mean']) for r in rr],yerr=[100*(float(r['upper'])-float(r['lower']))/2 for r in rr],fmt=mark,color=color,markersize=3.5,capsize=2,label=f'd={d}')
  es=np.linspace(0,.24,160);c=np.array([moments(e)['c'] for e in es]);y=100*(c/np.sqrt(1+2*c*c/g)/(8/(3*math.pi)*(1-.1/g))-1)
  ax.plot(es,y,'k--',lw=1,label='Frozen center')
  root=next(r for r in fc['roots'] if r['gamma']==g);ax.axvline(root['center_root'],color='k',lw=.7,alpha=.5)
  ax.axvline(.05015613022,color='#888888',lw=.7,ls=':',label='Large-gamma reference')
  ax.axhline(0,color='gray',lw=.7);ax.set_title(f'gamma = {g}');ax.set_xlabel('Dimensionless stabilizer e');ax.set_xlim(-.003,.245)
 axs[0].set_ylabel('Paired MSR advantage over Muon (pp)');axs[-1].legend(fontsize=7,loc='lower right')
 fig.suptitle('Fixed e grid; simultaneous seed t intervals; no fitted crossing',fontsize=11)
 fig.tight_layout();fig.savefig(OUT/'P1NUM_boundary.png');fig.savefig(OUT/'P1NUM_boundary.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,kind,margin in ((axs[0],'P2',.01),(axs[1],'P3',.005)):
  rr=[r for r in m if r['kind']==kind]
  labels=[]
  for i,r in enumerate(rr):
   mean=float(r['mean'])-float(r['target']);hw=(float(r['upper'])-float(r['lower']))/2
   ax.errorbar(100*mean,i,xerr=100*hw,fmt='o',color='#348ABD',capsize=3)
   name='Adam16' if r['name'].startswith('Adam') else 'Sign2 a='+('.2' if kind=='P3' else '1.5')
   labels.append(f"k={r['kappa']}, {name}, {r['metric']} (N={r['N']})")
  ax.axvspan(-100*margin,100*margin,color='green',alpha=.1);ax.axvline(0,color='gray',lw=.8)
  ax.set_yticks(range(len(rr)),labels,fontsize=8);ax.set_xlabel('Signed deviation from target (pp)');ax.set_title(f'{kind}: prespecified seed intervals');ax.invert_yaxis()
 fig.tight_layout();fig.savefig(OUT/'P1NUM_M_intervals.png');fig.savefig(OUT/'P1NUM_M_intervals.pdf');plt.close(fig)
 put(OUT/'P1NUM_plot_metadata.json',dict(resource_start=meta,resource_end=metadata(),inputs='Final CSV and frozen deterministic predictions; no new random data or fitted parameters'))
if __name__=='__main__':main()
