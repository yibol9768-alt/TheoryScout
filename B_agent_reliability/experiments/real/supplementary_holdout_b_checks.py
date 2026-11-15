"""Post-freeze offline guard checks; never sends HTTP or changes the frozen protocol."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import holdout_b as h
import reliability_core as d
import reliability_r2 as r

checks=[]
with tempfile.TemporaryDirectory(dir=h.ROOT) as tmp:
    out=Path(tmp);b=h.Budget(out,mock=True)
    for i in range(5): b.outcome(True,i+1)
    b.journal.f.close()
    resumed=h.Budget(out,mock=True,resume=True)
    assert resumed.fail_streak==0 and resumed.cooldown_until>r.time.monotonic()+590
    resumed.journal.f.close();checks.append("resume_preserves_cooldown_and_resets_streak")

class Response:
    def __init__(self,obj):self.obj=obj
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps(self.obj).encode()
class Opener:
    def __init__(self,obj):self.obj=obj
    def open(self,req,timeout):return Response(self.obj)
for mode in ("wrong_model","missing_usage","token_guard","non_string_content"):
    with tempfile.TemporaryDirectory(dir=h.ROOT) as tmp:
        out=Path(tmp);b=h.Budget(out,mock=True);trace=d.Journal(out/"sheet_d0_test.jsonl")
        obj=dict(model=d.MODEL,usage=dict(prompt_tokens=10,completion_tokens=10,total_tokens=20),
                 choices=[dict(finish_reason="stop",message=dict(content="{}"))])
        expected={"wrong_model":"model_mismatch","missing_usage":"missing_usage",
                  "token_guard":"token_guard","non_string_content":"invalid_response_shape"}[mode]
        if mode=="wrong_model":obj["model"]="FORBIDDEN_TEST_SENTINEL"
        elif mode=="missing_usage":obj.pop("usage")
        elif mode=="token_guard":obj["usage"]["completion_tokens"]=4097
        else:obj["choices"][0]["message"]["content"]=None
        with patch.dict(r.os.environ,{"SOLKEY":"TEST_SENTINEL_NOT_A_SECRET"}):
            c=r.Client(b);c.opener=Opener(obj)
            try:c.request([],trace,dict(step=0,review=0))
            except d.Stop as e:assert str(e)==expected
            else:raise AssertionError(mode)
        assert b.calls==1 and b.halt==expected
        trace.f.close();b.journal.f.close();checks.append(mode+"_halts_without_retry")

h.verify()
h.write(h.ROOT/"SUPPLEMENTARY_CHECKS_HOLDOUT_B.json",dict(utc=d.now(),checks=checks,passed=len(checks),
       real_calls=0,description="post-freeze offline checks; frozen code unchanged"))
print("SUPPLEMENTARY",len(checks),"PASS; zero real calls")
