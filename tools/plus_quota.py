"""Print Plus account quota as '5h_percent week_percent'. Self-contained copy for TheoryScout (no imports from other projects)."""
import glob, json, os, urllib.request
f = glob.glob(os.path.expanduser("~/.local/share/codex-workflow/llm-proxy-private/auth-plus/*plus*.json"))[0]
t = json.load(open(f)); t = t.get("tokens", t)
req = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
    "Authorization": "Bearer " + t["access_token"], "ChatGPT-Account-Id": t["account_id"], "User-Agent": "codex_cli_rs"})
w = json.loads(urllib.request.urlopen(req, timeout=40).read())["rate_limit"]
print(w["primary_window"]["used_percent"], w["secondary_window"]["used_percent"])
