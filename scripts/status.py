import json, glob, os
for f in sorted(glob.glob(os.path.expanduser("~/runs/*/log.jsonl"))):
    if "smoke" in f: continue
    r = [json.loads(l) for l in open(f)][-1]
    n = f.split("/")[-2]
    print(n, "step", r["step"], "tok %.0fM" % (r["tok"]/1e6), "l0 %.0f" % r["l0"], "lam %.3g" % r["lam"],
          "fvu", [round(x, 2) for x in r["fvu"]], "l0L", [round(x) for x in r["l0_layer"]], "sec", r["sec"])
