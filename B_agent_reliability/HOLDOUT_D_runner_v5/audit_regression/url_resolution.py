from unittest.mock import patch
from common import *
guards();sys.path.insert(0,str(RUNNER))
from transport import HTTPTransport,TestHTTPTransport,URL
results=[]
for url in (URL,'http://127.0.0.1:18319/anything','http://127.0.0.1:18319@audit.invalid/v1/chat/completions','http://127.0.0.1:18319@audit.invalid:80/v1/chat/completions'):
    for cls in (HTTPTransport,TestHTTPTransport):
        with patch('http.client.HTTPConnection.connect',side_effect=AssertionError('CONNECT_REACHED')),patch('urllib.request.build_opener',side_effect=AssertionError('CLIENT_REACHED')):
            try:cls(live=True,test_mock_url=url) if cls is HTTPTransport else cls(url)
            except ValueError as ex:results.append(dict(url=url,transport=cls.__name__,rejected=str(ex),connection_targets=[]))
            else:raise AssertionError('dangerous address accepted')
save('url_resolution.json',dict(resources=resources(),results=results))
