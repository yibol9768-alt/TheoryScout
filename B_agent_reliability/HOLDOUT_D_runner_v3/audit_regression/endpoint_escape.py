from unittest.mock import patch
from common import *
guards();engine=modules()
from transport import HTTPTransport,MockServer,URL
from evidence import FIRST,RELEASE
def good():return dict(five_hour=0.,week=0.)
p=OUT/'endpoint_escape';e=engine.Engine(p,verify=False);server=MockServer(policy='fatal_at_8')
for seed in FIRST:
    while not e.tasks[seed].terminal:e.request(seed,server,good)
states=[]
for restart in (False,True):
    if restart:e=engine.Engine(p,verify=False)
    before=e.attempts
    with patch('urllib.request.build_opener',side_effect=AssertionError('NETWORK_CLIENT_CONSTRUCTED')):
        try:
            tr=HTTPTransport(live=True,test_mock_url=URL)
            e.request(FIRST[0]+2,tr,good)
        except ValueError as ex:rejected=str(ex)
        else:raise AssertionError('endpoint escape')
    assert e.attempts==before and e.tasks[FIRST[0]+2].step==0 and not e.smoke
    states.append(dict(restart=restart,rejected=rejected,third_c=0,smoke=e.smoke,attempts=e.attempts,mode=e.mode))
    e.close()
save('endpoint_escape.json',dict(resources=resources(),opens=[],states=states,release_exists=RELEASE.exists()))
