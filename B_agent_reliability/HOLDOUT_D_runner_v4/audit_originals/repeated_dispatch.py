from unittest.mock import patch
from common import *
guards();engine=modules()
from transport import HTTPTransport,MockServer,URL
from protocol import Task
import evidence
e=engine.Engine(OUT/'repeat_dispatch',verify=False);opens=[]
class Response:
    status=200
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self):return MockServer().send(Task(evidence.FIRST[0]).body())[1]
class AtOpen:
    def open(self,req,timeout):opens.append(dict(url=req.full_url,body_sha256=digest(req.data)));return Response()
# Project only the authorization-file check. Do not alter reserve or send logic.
tr=object.__new__(HTTPTransport);tr.testing=False;tr.url=URL;tr.opener=AtOpen()
with patch.object(engine.Engine,'mode',new=property(lambda self:'live')),patch('engine.require_live_release',return_value=None),patch('transport.require_live_release',return_value=None),patch.dict(os.environ,{'SOLKEY':'AUDIT_DUMMY_NOT_SECRET'}):
    reservation,body=e.reserve(evidence.FIRST[0],lambda:dict(five_hour=0.,week=0.))
    for _ in range(3):tr.send(body,engine=e,reservation=reservation)
    result=dict(opens=opens,attempts=e.attempts,successful=e.success,pending=len(e.pending),active=len(e.active),reservation_call=reservation['call'])
    assert len(opens)==3 and e.attempts==1 and len(e.pending)==1
e.close()
save('repeated_dispatch.json',dict(resources=resources(),result=result,projection='Missing production release check only; no real response, no socket, no release file. Same unmodified send method reuses one valid reservation.'))
print(json.dumps(result))
