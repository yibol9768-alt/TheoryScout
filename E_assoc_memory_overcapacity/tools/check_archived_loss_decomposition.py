"""Reconstruct R24's finite-count partitions and grid-loss budgets."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];r=read(args.input/'RESULT.json');old=read(root/'revisions/r8_random_thresholds/results.json')
    for name,digest in r['source_hashes'].items():assert sha(root/name)==digest
    rows=[];count_checks=0
    for row in r['rows']:
        length=row['L'];ref=next(v for v in old['rows'] if v['L']==length)
        assert row['paired']['event_counts']['success']==ref['references']['actual']['count']
        assert row['R8_fixed_b_lower_paired_count']==ref['references']['lower_quadratic']['count']
        for label in ['paired','all_pairs']:
            part=row[label];c=part['event_counts'];s=c['success'];e=c['energy_only_tangent'];center=c['center_only_tangent'];both=c['double_tangent'];den=part['denominator']
            cats=part['loss_categories'];single=e-cats['center_alone_only']
            assert 0<=both<=single<=min(e,center)<=s<=den
            assert cats['energy_alone_only']==center-single and cats['either_alone']==s-e-center+single
            assert cats['joint_only']==single-both and sum(cats.values())==s-both
            assert F(part['loss_rate'])==F(s-both,den)
            assert F(part['center_two_order_allocation'])==F(s-center+e-both,2*den)
            assert F(part['energy_two_order_allocation'])==F(s-e+center-both,2*den)
            count_checks+=1
        part=row['all_pairs'];c=part['event_counts'];den=r['sample_queries']*r['sample_thresholds'];assert part['denominator']==den
        g18=row['grid18']['lower_numerator'];g258=row['grid258']['lower_numerator']
        assert g18<=g258<=c['double_tangent']<=row['grid258']['upper_numerator']<=row['grid18']['upper_numerator']
        terms=row['loss_to_empirical_grid18_numerators']
        assert terms==dict(tangent=c['success']-c['double_tangent'],remaining_fine_mesh=c['double_tangent']-g258,coarse_to_fine_mesh=g258-g18)
        assert sum(terms.values())==c['success']-g18
        assert all(F(row['loss_to_empirical_grid18_rates'][k])==F(v,den) for k,v in terms.items())
        old_loss=F(c['success']-row['R8_fixed_b_lower_all_pairs_count'],den)
        assert old_loss==F(row['R8_fixed_b_tangent_loss_rate']) and old_loss>=0
        rows.append(dict(L=length,center_allocation_pp=float(100*F(part['center_two_order_allocation'])),
            energy_allocation_pp=float(100*F(part['energy_two_order_allocation'])),
            double_tangent_loss_pp=float(100*F(part['loss_rate'])),fixed_b_tangent_loss_pp=float(100*old_loss),
            grid18_to_grid258_loss_pp=float(F(terms['coarse_to_fine_mesh']*100,den)),
            fine_mesh_loss_pp=float(F(terms['remaining_fine_mesh']*100,den))))
    v=dict(partition_and_allocation_reconstructions=count_checks,grid_budget_reconstructions=len(rows),
           original_R8_pair_counts_recovered=True,rows=rows,source_result_sha256=sha(args.input/'RESULT.json'),
           interpretation='Exact finite sums on archived floats, NOT a C0 certificate or a confidence interval')
    with (args.out/'CHECKS.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps(v,indent=2))


if __name__=='__main__':main()
