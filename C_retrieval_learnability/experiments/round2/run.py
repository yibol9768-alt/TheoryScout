"""Resource-bounded supervisor and two workers. No external services."""
from core import *
import subprocess
from datetime import datetime, timezone


def main():
    decision = json.loads((OUT/'budget_decision.json').read_text())
    T = decision['steps']
    if len(sys.argv)>1:
        worker = int(sys.argv[1])
        deadline = float(sys.argv[2])
        configs = json.loads((OUT/'manifest.json').read_text())['configs']
        for index, cfg in enumerate(configs):
            if index%2!=worker:
                continue
            if time.time()>deadline:
                break
            identifier = f'n{cfg["n"]}_k{cfg["k"]}_d{cfg["d"]}_s{cfg["seed"]}_{cfg["loss"]}'
            result = run(cfg, T, deadline)
            save(OUT/'raw'/f'{identifier}.json', result)
            print(json.dumps(dict(id=identifier, status=result['status'],
                                  seconds=result['elapsed_seconds'])), flush=True)
        return
    configs = [dict(n=n, k=k, d=d, seed=seed, loss=loss) for n,k,d in SETTINGS
               for seed in range(10,15) for loss in LOSSES]
    files = [ROOT/'PREREG_ROUND2.md', OUT/'core.py', OUT/'prepare.py', OUT/'run.py',
             OUT/'budget_decision.json', OUT/'validation.json', ROOT/'experiments/week1_common.py',
             ROOT/'FORMULATION.md', ROOT/'RESULTS_WEEK1.md', ROOT.parent/'docs/codex_week1_audit.txt']
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in files}
    start = time.time()
    hard_end = decision['preparation_start_epoch']+7000
    deadline = min(start+6000, hard_end-120)
    save(OUT/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                  configs=configs, steps=T, workers=2, hashes=hashes,
                                  deadline_epoch=deadline, interpreter=sys.executable,
                                  numpy_version=np.__version__, env={k:os.environ[k] for k in
                                  ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    processes, logs = [], []
    for worker in range(2):
        log = (OUT/f'worker{worker}.log').open('w')
        logs.append(log)
        processes.append(subprocess.Popen([sys.executable, str(OUT/'run.py'), str(worker), str(deadline)],
                                           stdout=log, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        if time.time()>deadline+30:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            break
        time.sleep(1)
    codes = [p.wait() for p in processes]
    for log in logs:
        log.close()
    save(OUT/'execution.json', dict(start_epoch=start, end_epoch=time.time(),
                                   wall_seconds=time.time()-start, returncodes=codes,
                                   raw_files=len(list((OUT/'raw').glob('*.json'))), memory=memory()))
    print(json.dumps(json.loads((OUT/'execution.json').read_text())), flush=True)


if __name__ == '__main__':
    main()
