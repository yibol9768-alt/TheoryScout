"""Standard-library C0 joint-pencil audit; native and integral replays optional."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import json

import certify_weighted_chisquare as integer
from check_anchor_certificate import construction_arrays,repeat
from check_spectral_certificate import binary_arrays,read,sha
from check_recall_upper import audit_integral


def references():
    matrices=[[[F(1),F(0)],[F(0),F(1)]],[[F(1),F(1,2)],[-F(1,4),F(3,4)]],[[F(0),F(1)],[F(-1),F(0)]]]
    us=[(F(1),F(0)),(F(0),F(1)),(F(3,5),F(4,5)),(F(3,5),-F(4,5))]
    identities=perturbations=0
    for b,u,n,s,c in product(matrices,us,[4,7],[F(1,4),F(2,3)],[F(1,4),F(3,10),F(1,2)]):
        bu=[sum(x*y for x,y in zip(row,u)) for row in b];a=sum(x*y for x,y in zip(u,bu));v=sum(x*x for x in bu)
        D=n*s*a;V=F(n*n,n-1)*(s*v-s*s*a*a)
        assert n*c*D-D*D-(n-1)*V==n*n*s*(c*a-v);identities+=1
        for e in [F(1,1024),F(1,4096)]:
            err=[[e,-e],[F(0),e]];eta=sum(abs(x) for row in err for x in row);norm=sum(abs(x) for row in b for x in row)
            nu=2*norm*eta+eta*eta
            def pencil(matrix):
                return [[c*(matrix[i][j]+matrix[j][i])/2-sum(matrix[k][i]*matrix[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
            old=pencil(b);new=pencil([[b[i][j]+err[i][j] for j in range(2)] for i in range(2)])
            assert max(sum(abs(new[i][j]-old[i][j]) for j in range(2)) for i in range(2))<=c*eta+nu;perturbations+=1
    return dict(exact_joint_event_identity_cases=identities,exact_operator_transfer_reference_cases=perturbations)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--repeat-kernels',action='store_true')
    parser.add_argument('--reintegrate',action='store_true');args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];directory=args.certificate.resolve();result=read(directory/'RESULT.json');design=read(directory/'DESIGN.json')
    assert result['source_hashes']==design['source_hashes']
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    hist=read(root/'revisions/R27_HISTORY.json');manifest=read(root/hist['path']/'manifest.json')
    assert sha(root/'tools/check_recall_upper.py')==manifest['files']['tools/check_recall_upper.py']
    for stem in ['rounding_anchor_construction','rounding_spectral_residual']:
        build=read(directory/(stem+'_BUILD.json'))
        assert sha(directory/stem)==build['executable_sha256'] and sha(root/'tools'/(stem+'.cpp'))==build['source_sha256']
    inputs=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json');n=inputs['n'];q=inputs['q'];eta=F(inputs['eta']);nu=F(inputs['N_operator_perturbation'])
    assert (n,q)==(design['n'],design['q'])==(result['n'],result['q'])
    assert (eta,nu)==(F(design['eta_K']),F(design['nu_N']))
    original=(root/'revisions/r11_anchor_certificate_v2/actual_construction.bin').read_bytes();audits=[];widths=[]
    assert [r['c'] for r in result['rows']]==design['coefficients']==['1/4','3/10']
    for row in result['rows']:
        node=directory/row['directory'];out=args.out/row['directory'];out.mkdir();c=F(row['c'])
        construction=read(node/'CONSTRUCTION.json');path=node/'actual_construction.bin'
        assert sha(path)==construction['input_sha256']
        qc,b,h,scalars=construction_arrays(path);assert qc==q
        assert path.read_bytes()[8:8+8*q*q]==original[8:8+8*q*q]
        assert all(abs(v)<=2.**200 for array in (b,h,scalars) for v in array)
        assert all(h[i*q+j]==h[j*q+i] for i in range(q) for j in range(q))
        aa,dd=F(construction['scalar_A']),F(construction['scalar_D']);la,ha,ld,hd=map(F,scalars)
        assert (aa,dd)==(c,F(1)) and la<=c<=ha and ld<=1<=hd
        error_c=q*F.from_float(float.fromhex(construction['rounding_output']['max_entry_error_hex']))
        assert error_c==F(construction['operator_error'])
        spectrum=read(node/'actual_spectrum.json');path_s=node/'actual.bin';assert sha(path_s)==spectrum['input_sha256']
        qs,hs,basis,eigen=binary_arrays(path_s);assert qs==q and h==hs and list(eigen)==sorted(eigen)
        assert all(abs(v)<=2.**200 for array in (hs,basis,eigen) for v in array)
        eigen=list(map(F,eigen));assert eigen==list(map(F,spectrum['eigenvalues']))
        fields=spectrum['rounding_output'];gram=q*F.from_float(float.fromhex(fields['max_gram_hex']));rho=q*F.from_float(float.fromhex(fields['max_residual_hex']))
        assert gram==F(spectrum['gram_operator_bound'])<1 and rho==F(spectrum['residual_operator_bound'])
        inverse=1/(1-integer.I(gram)).sqrt();bound=integer.I(rho)*inverse+2*integer.I(max(map(abs,eigen)))*(1+integer.I(gram)).sqrt()*(inverse-1)
        assert spectrum['eigenvalue_error']==bound.record();error_s=F(bound.hi,integer.SCALE)
        assert error_s==F(spectrum['epsilon_upper'])
        errors=read(node/'ERRORS.json');coupling=c*eta+nu;total=error_c+error_s+coupling
        assert errors==dict(construction=str(error_c),spectral=str(error_s),C0_operator_transfer=str(coupling),total_C0_eigenvalue_error=str(total))
        assert total==F(row['C0_eigenvalue_error']) and eigen[0]+total<0<eigen[-1]-total
        if args.repeat_kernels:
            repeat(directory,out,'rounding_anchor_construction',path,construction['rounding_output'])
            repeat(directory,out,'rounding_spectral_residual',path_s,fields)
        interval=[]
        for side,sign in [('lower',-1),('upper',1)]:
            values=[v+sign*total for v in eigen];filename='probability_'+side+'.json';saved=read(node/filename)
            upper,width=audit_integral(node,filename,values,[1]*q);widths.append(width)
            if args.reintegrate:
                replay,_=integer.certify(values,[1]*q,integer.pi_interval(),integer.gauss8(),target=F(saved['target_width']))
                assert replay==saved
                with (out/filename).open('x') as f:json.dump(replay,f,indent=2);f.write('\n')
            interval.append(max(F(0),F(int(saved['probability']['lower_integer']),integer.SCALE)) if side=='lower' else upper)
        assert row['probability_C0_positive']==list(map(str,interval))
        audits.append(dict(c=str(c),C0_probability_interval=list(map(str,interval)),C0_transfer=str(coupling),native_eigenvalue_error=str(error_c+error_s),total_C0_error=str(total)))
    assert F(audits[0]['C0_probability_interval'][1])<=F(audits[1]['C0_probability_interval'][0])
    witness_path=root/'revisions/r39_moment_witnesses_v4/RESULT.json';hist=read(root/'revisions/R39_HISTORY.json');manifest=read(root/hist['path']/'manifest.json')
    assert sha(witness_path)==manifest['files']['revisions/r39_moment_witnesses_v4/RESULT.json']
    cuts=[]
    for row in read(witness_path)['rows']:
        for witness in row['witnesses']:
            for audit in audits:
                c=F(audit['c']);lo,hi=map(F,audit['C0_probability_interval'])
                probability=sum(F(a['probability']) for a in witness['atoms'] if n*c*F(a['D'])-F(a['D'])**2-(n-1)*F(a['sqrt_V'])**2>0)
                cuts.append(dict(L=row['L'],direction=witness['direction'],c=str(c),witness_joint_event_probability=str(probability),
                    excluded_by_new_C0_joint_probability=probability<lo or probability>hi))
    output=dict(reference=references(),source_hashes=len(result['source_hashes']),matrix_and_C0_transfer_audits=audits,
        native_kernel_repetitions=4 if args.repeat_kernels else 0,integer_integrals_recomputed=4 if args.reintegrate else 0,
        saved_integral_accounting_audits=4,max_comparison_integral_width_approximate=float(max(widths)),
        R39_scalar_law_joint_event_checks=cuts,old_scalar_laws_tested=4,
        not_full_energy_flip_or_recall_probability=True,independent_proof_review=False)
    with (args.out/'CHECKS.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(dict(audits=audits,witness_joint_checks=cuts),indent=2))


if __name__=='__main__':main()
