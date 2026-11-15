"""Check R21 propagation and independent high-precision arithmetic diagnostics.

Decimal cross-checks are diagnostics, not a replacement for interval proofs.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D,localcontext
import argparse
import hashlib
import json

import certify_sphere_threshold_cdf as sphere


def read(path):return json.loads(path.read_text())
def decimal(value):
    value=F(value);return D(value.numerator)/D(value.denominator)
def endpoints(record):
    assert record['denominator_power2']==160
    return D(record['lower_integer'])/D(sphere.S),D(record['upper_integer'])/D(sphere.S)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    result=read(args.input/'RESULT.json');radial=read(args.input/'RADII.json');m=result['m']
    for path,digest in result['source_hashes'].items():assert sphere.sha(root/path)==digest
    radius_map={(r['tail'],F(r['radius'])):r for r in radial['records']}
    radial_checks=[];normal_checks=[]
    with localcontext() as context:
        context.prec=110
        for r in radial['records']:
            rho=F(r['radius']);x=decimal(F(m,2)*rho*rho)
            # Separate arithmetic implementation, direct Decimal.exp, no range reduction.
            term=total=D(1)
            for k in range(1,m//2):term=term*x/D(k);total+=term
            upper=total/x.exp();reference=1-upper if r['tail']=='lower' else upper
            lo,hi=endpoints(r['probability']);assert lo<=reference<=hi
            radial_checks.append(dict(radius=r['radius'],tail=r['tail'],reference=str(reference),inside_interval=True))
        pi=sphere.integer.pi_interval();dpi=(D(pi.lo)+D(pi.hi))/(2*D(sphere.S))
        arguments={F(0),F(1,2),F(1),F(2),F(4),F(6),F(7)}
        for row in result['cdf_enclosures']:
            for node in row['nodes']:
                for key in ['lower_radius','upper_radius']:
                    if node[key] is not None:arguments.add(F(node['t'])*F(node[key]))
        for argument in sorted(arguments):
            x=decimal(argument);term=total=x;k=0
            # Positive expansion of the Gaussian integral, rather than alternating Taylor.
            while term:
                term=term*x*x/D(2*k+3);total+=term;k+=1
                if abs(term)<D('1e-105'):break
                assert k<1000
            reference=D(1)/2+(-x*x/2).exp()*total/(2*dpi).sqrt()
            interval=sphere.normal_cdf(argument,pi);lo,hi=endpoints(interval.record())
            assert lo<=reference<=hi
            normal_checks.append(dict(x=str(argument),reference=str(reference),inside_interval=True))
    # Reconstruct each selected CDF endpoint from saved radial bounds and exact t.
    for row in result['cdf_enclosures']:
        length=row['L']
        for node in row['nodes']:
            t=F(node['t']);lo=F(node['lower']);hi=F(node['upper']);assert 0<=lo<=hi<=1
            for label,key in [('lower','lower_radius'),('upper','upper_radius')]:
                if node[key] is None:continue
                rho=F(node[key]);rr=radius_map[label,rho]['probability'];delta=F(int(rr['upper_integer']),sphere.S)
                h=sphere.gaussian_order_cdf(m,length,t*rho,pi)
                if label=='lower':
                    expected=max(F(0),F(((h-sphere.I(delta))/(1-sphere.I(delta))).lo,sphere.S));assert expected==lo
                else:
                    expected=min(F(1),F((h/(1-sphere.I(delta))).hi,sphere.S));assert expected==hi
    point=read(root/'revisions/r20_threshold_family/RESULT.json')['certified_nodes']
    ts=sorted({F(v['t']) for v in point});heights=[max(F(v['target_probability_lower']) for v in point if F(v['t'])==t) for t in ts]
    for j in range(len(ts)-2,-1,-1):heights[j]=max(heights[j],heights[j+1])
    assert all(0<=h<=1 for h in heights);heights.append(F(0))
    for row,recall in zip(result['cdf_enclosures'],result['conditional_recall_lower']):
        lower=[F(v['lower']) for v in row['nodes']]
        for j in range(1,len(ts)):lower[j]=max(lower[j],lower[j-1])
        assert lower==list(map(F,row['monotone_lower']))
        bound=sum((heights[j]-heights[j+1])*lower[j] for j in range(len(ts)))
        assert bound==F(recall['conditional_recall_lower'])
    for node,target in zip(result['circle_reference_checks'],[F(1,2),F(1)]):
        assert F(node['lower'])<=target<=F(node['upper'])
    save=dict(radial_decimal_checks=radial_checks,normal_positive_series_checks=normal_checks,
              decimal_precision=110,decimal_checks_are_diagnostics=True,cdf_endpoint_reconstructions=36,
              exact_recall_jump_reconstructions=2,reads_threshold_samples=False,
              source_result_sha256=sphere.sha(args.input/'RESULT.json'),
              scope='Numeric checks of R21; not independent proof review')
    with (args.out/'CHECKS.json').open('x') as stream:json.dump(save,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(radial_checks=len(radial_checks),normal_checks=len(normal_checks),
                          cdf_endpoints=36,recall_sums=2,status='PASS'),indent=2))


if __name__=='__main__':main()
