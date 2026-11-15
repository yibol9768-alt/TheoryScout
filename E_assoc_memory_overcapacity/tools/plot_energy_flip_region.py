"""Display exact flip-region vertices; floating conversion is for plotting only."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Rectangle
    rows=json.loads((a.certificate/'RESULT.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.5),layout='constrained')
    for ax,row in zip(axes,rows):
        vertices=[(100*float(F(p)),100*float(F(n))) for p,n in row['joint_outer_polygon']['vertices']]
        cp=sum(v[0] for v in vertices)/len(vertices);cn=sum(v[1] for v in vertices)/len(vertices)
        vertices.sort(key=lambda v:math.atan2(v[1]-cn,v[0]-cp))
        pu=100*float(F(row['components']['positive']['interval'][1]));nu=100*float(F(row['components']['negative']['interval'][1]))
        ax.add_patch(Rectangle((0,0),pu,nu,fill=False,linestyle='--',edgecolor='#8a8a8a',linewidth=1.2,label='Separate component bounds'))
        ax.add_patch(Polygon(vertices,closed=True,facecolor='#c4dcf0',edgecolor='#205a87',linewidth=1.6,label='Joint outer region'))
        lim=max(pu,nu)*1.08;ax.plot([0,lim],[0,lim],color='#666666',linestyle=':',linewidth=1.2,label='Zero net effect')
        ax.set(xlim=(0,lim),ylim=(0,lim),aspect='equal',title=f"top{row['L']}",xlabel='Positive-direction flips (pp)',ylabel='Negative-direction flips (pp)')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper right',fontsize=7)
    fig.suptitle('Shared-threshold energy flips, conditional on represented C0',fontsize=12)
    fig.supxlabel('Outer enclosures under inherited certificates; vertices need not be attainable.',fontsize=9)
    for suffix in ['png','pdf']:
        target=a.out.with_suffix('.'+suffix)
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
