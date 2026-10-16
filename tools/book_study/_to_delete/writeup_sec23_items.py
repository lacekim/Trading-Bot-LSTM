"""Turns sec23.json into replacement items for Equations 1-15 and Expression 1 (read by writeup_build as items_patch_*.json)."""
import json
import writeup_core as W
S = json.load(open(W.OUT / "sec23.json"))
items = {i["id"]: i for i in json.load(open(W.OUT / "items_ch01_07.json"))}
NAMES = {"TIB": "tick imbalance", "VIB": "volume imbalance", "DIB": "dollar imbalance", "TRB": "tick run", "VRB": "volume run", "DRB": "dollar run"}
ASSUME = ("Built from our hourly reference rows (Kraken ETH, Alpaca AAPL; the finest rows on both venues in the workspace), not ticks: each row is one 'tick', b_t is Equation 1 on the row closes, v_t is the row volume "
          "(dollar bars: close x volume). The book leaves open (assumptions): E0[T] starts at 24 rows, the expectations are exponentially weighted over bars with alpha = 2/101, the first 24 rows are warm-up, the first b_t is +1, "
          "and a bar may run across a market close. Nothing was tuned.")

def table(kinds):
    rows = []
    for v in ("Kraken", "Alpaca"):
        for k in kinds:
            r = S[f"{v}|{k}"]
            rows.append([v, k, f"{r['rows']:,}", f"{r['bars']:,}", f"{r['rows_per_bar']:.2f}", f"{r['median']:.0f}", f"{100*r['one_row']:.1f}%", r["max_T"], r["check"], "yes" if r["et_ok"] else "NO"])
    return W.tbl(["Venue", "Bars", "Rows used", "Bars built", "Mean rows per bar", "Median", "One-row bars", "Longest (rows)", "Closing rule holds (bars)", "E0[T] recomputed"], rows)

DEF = {2: ("TIB",), 3: ("TIB",), 4: ("VIB", "DIB"), 5: ("VIB", "DIB"), 6: ("VIB", "DIB"), 7: ("VIB", "DIB"), 8: ("VIB", "DIB"), 9: ("VIB", "DIB"), 10: ("TRB",), 11: ("TRB",), 12: ("TRB",), 13: ("VRB", "DRB"), 14: ("VRB", "DRB"), 15: ("VRB", "DRB")}
SHOW = {3, 9, 12, 15}
out = []
for n, kinds in DEF.items():
    it = dict(items[f"Equation {n}"])
    base = (it.get("text") or "").split("Status:")[0].strip()
    it["basis"] = "data"
    names = " and ".join(NAMES[k] for k in kinds)
    if n in SHOW:
        it["text"] = base + f"  Built here as {names} bars (Equations {min(k for k,v in DEF.items() if v==kinds)}-{n}); the table counts the bars. " + ASSUME
        it["table"] = table(kinds)
        it["note"] = ("Closing rule holds = bars whose theta reaches the threshold on the last row and not before, replayed independently from the stored thresholds; E0[T] recomputed = the expected bar length rebuilt from the stored bar lengths. "
                      "The bar counts depend on the unspecified choices above: with hourly rows the expectation feedback can shrink or stretch the bars (for example Alpaca tick and volume imbalance bars collapse to one row).")
    else:
        it["text"] = base + f"  Used in the {names} bars (see Equations {max(k for k in SHOW if k>=n)} for their counts)."
        it["table"] = None
    out.append(it)
e1 = dict(items["Equation 1"]); k = S["Kraken|TIB"]["buys"]; a = S["Alpaca|TIB"]["buys"]
e1["text"] = f"b_t on the closes of our hourly reference rows (b_t carried forward when the close does not change). Share of rows with b_t = +1: Kraken ETH {100*k:.1f}%, Alpaca AAPL {100*a:.1f}%. " + ASSUME
out.append(e1)
ex = dict(items["Expression 1"]); ex["text"] = "The factor inside Equations 14 and 15; it is the threshold factor of the volume and dollar run bars built in Equation 15's entry."; out.append(ex)
json.dump(out, open(W.OUT / "items_patch_sec23.json", "w"), indent=1, default=str)
print(len(out), "patched items")
