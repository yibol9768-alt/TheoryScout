"""Test the extracted bundle with network disabled and original workspace hidden.

Run only inside unshare -Urnm. All mounts are private to that child namespace.
This helper is research test evidence, not needed for a normal bundle replay.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--empty',type=Path,required=True)
parser.add_argument('--parent-mount-ns',required=True)
parser.add_argument('--parent-net-ns',required=True)
args=parser.parse_args()
mount_ns=os.readlink('/proc/self/ns/mnt');net_ns=os.readlink('/proc/self/ns/net')
assert mount_ns!=args.parent_mount_ns and net_ns!=args.parent_net_ns
assert args.bundle.resolve().is_relative_to(Path('/tmp'))
assert args.empty.is_dir() and not list(args.empty.iterdir())
subprocess.run(['mount','--make-rprivate','/'],check=True)
subprocess.run(['mount','--bind',str(args.empty),str(args.workspace)],check=True)
assert not list(args.workspace.iterdir()),'Original workspace is still accessible'
with args.out.with_suffix('.log').open('x') as log:
    subprocess.run([sys.executable,'-B',str(args.bundle/'reproduce.py'),'--out',str(args.out)],
                   cwd='/tmp',stdout=log,stderr=subprocess.STDOUT,check=True)
record=dict(parent_mount_namespace=args.parent_mount_ns,child_mount_namespace=mount_ns,
            parent_network_namespace=args.parent_net_ns,child_network_namespace=net_ns,
            original_workspace_hidden=True,network_namespace_isolated=True,
            original_workspace=str(args.workspace),extracted_bundle=str(args.bundle),output=str(args.out),
            replay=json.loads((args.out/'RESULT.json').read_text()))
args.out.with_suffix('.isolation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
