import sys
sys.path.insert(0,'/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/TheoryScout/B_agent_reliability/HOLDOUT_D_runner_v5/amend5_checks')
import harness
harness.install_common(__import__("pathlib").Path(__file__).parent)
exec(compile(harness.ROOT.joinpath("audit_d3_checks/entry_sentinel.py").read_text(),"entry_sentinel.py","exec"))
