"""Print the layers of an Ollama registry manifest read on stdin: media type, size, digest.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14
"""
import json,sys
d=json.load(sys.stdin)
for l in d.get("layers",[]):
    mt=l["mediaType"].replace("application/vnd.ollama.image.","")
    print("  %-14s %12d  %s" % (mt, l["size"], l["digest"][7:19]))
