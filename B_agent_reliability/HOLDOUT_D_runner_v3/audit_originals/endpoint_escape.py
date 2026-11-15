from unittest.mock import patch
from common import *
guards();engine=modules()
from transport import HTTPTransport,MockServer,URL
from evidence import FIRST,RELEASE
def good():return dict(five_hour=0.,week=0.)
p=OUT/'endpoint_escape';e=engine.Engine(p,verify=False);server=MockServer(policy='fatal_at_8')
for seed in FIRST:
    while not e.tasks[seed].terminal:e.request(seed,server,good)
opens=[]
class FakeResponse:
    status=200
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self):return MockServer().send(self.body)[1]
class AtOpen:
    def open(self,req,timeout):
        opens.append(dict(url=req.full_url,step=json.loads(json.loads(req.data)['messages'][1]['content'])['step'],body_sha256=digest(req.data)))
        return FakeResponse(req.data)
states=[]
for restart in (False,True):
    if restart:e=engine.Engine(p,verify=False)
    with patch('urllib.request.build_opener',return_value=AtOpen()):
        transport=HTTPTransport(live=True,test_mock_url=URL)
        e.request(FIRST[0]+2,transport,good)
    states.append(dict(restart=restart,third_c=e.tasks[FIRST[0]+2].step,smoke=e.smoke,attempts=e.attempts,mode=e.mode))
    e.close()
assert len(opens)==2 and all(z['url']==URL for z in opens) and all(not x['smoke'] for x in states)
save('endpoint_escape.json',dict(resources=resources(),opens=opens,states=states,release_exists=RELEASE.exists(),
    patch='Only urllib opener replaced with no-network sentinel; release/smoke/budget/request logic untouched. Output directory remapped.'))
print(json.dumps(dict(opens=opens,states=states),ensure_ascii=False))
