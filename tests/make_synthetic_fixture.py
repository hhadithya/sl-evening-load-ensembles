"""Synthetic edge-case fixture with known answers (NOT real data) for unit-testing the audit rules.
Writes synthetic_data.zip mirroring the Zenodo member paths, and expected.json."""
import io, json, zipfile, sys
H = ("household_ID,timestamp,date,time,obis,importkwh(kwh),exportkwh(kwh),importkvarh(kvarh),exportkvarh(kvarh),powerfactor,"
     "avgcurrent(v),avgexportkw(kw),avgimportkw(kw),avgvoltage(v),phaseainstcurrent(a),phaseainstvoltage(v),phasebinstcurrent(a),"
     "phasebinstvoltage(v),phasecinstcurrent(a),phasecinstvoltage(v)")
import datetime as dt
def ts(d, hh, mm):
    utc = dt.datetime.fromisoformat(f"{d}T{hh:02d}:{mm:02d}:00") - dt.timedelta(hours=5, minutes=30)
    ms = int(utc.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    return f"{float(round(ms, -7)):.1f}"
def row(h, d, hh, mm, imp, exp=0.0, ai=None):
    ai = "" if ai is None else f"{ai}"
    return f"{h},{ts(d,hh,mm)},{d},{hh:02d}:{mm:02d}:00,LP,{imp},{exp},,,0.9,,0.0,{ai},230,,,,,,"
rows = []
def day(h, d, start, per15, skip=(), dup_exact=(), dup_conf=(), jump=None):
    R = start
    for s in range(96):
        hh, mm = divmod(s * 15, 60)
        if s > 0: R = round(R + per15, 4)
        if jump and s == jump[0]: R = round(R + jump[1], 4)
        if s in skip: continue
        rows.append(row(h, d, hh, mm, R, ai=round(per15 * 4, 4)))
        if s in dup_exact: rows.append(row(h, d, hh, mm, R, ai=round(per15 * 4, 4)))
        if s in dup_conf: rows.append(row(h, d, hh, mm, round(R + 3, 4), ai=round(per15 * 4, 4)))
    return R
# A: complete day, 0.05 kWh per 15 min -> peak 16*0.05 = 0.8
day("IDA", "2024-11-01", 100.0, 0.05)
# B: missing 20:00 (slot 80) -> missing_reading
day("IDB", "2024-11-01", 200.0, 0.05, skip={80})
# C: exact duplicate at 19:00 (slot 76) -> still valid, peak 0.8; plus missing 18:30 is NOT tested here
day("IDC", "2024-11-01", 300.0, 0.05, dup_exact={76})
# D: conflicting duplicate at 21:00 (slot 84) -> conflicting_duplicate
day("IDD", "2024-11-01", 400.0, 0.05, dup_conf={84})
# E: register drops by 50 at 19:30 (slot 78) -> negative_interval
day("IDE", "2024-11-01", 500.0, 0.05, jump=(78, -50))
# F: +30 kWh jump at 20:30 (slot 82) -> implausible_interval (cap 10)
day("IDF", "2024-11-01", 600.0, 0.05, jump=(82, 30))
# G: missing 22:30 only (slot 90, the closing reading) -> missing_reading
day("IDG", "2024-11-01", 700.0, 0.05, skip={90})
# H: missing 18:15 and 22:45 (outside window) -> valid, peak 0.8
day("IDH", "2024-11-01", 800.0, 0.05, skip={73, 91})
# off-grid row (excluded and counted)
rows.append(row("IDA", "2024-11-02", 10, 7, 999.0))
buf = io.StringIO(); buf.write(H + "\n" + "\n".join(rows) + "\n")
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("data/consumption_data/smart_meter/15min_interval/smart_15min_1.csv", buf.getvalue())
    z.writestr("data/consumption_data/smart_meter/15min_interval/smart_15min_2.csv", H + "\n")
    z.writestr("data/consumption_data/smart_meter/15min_interval/smart_15min_3.csv", H + "\n")
json.dump({"valid": 3, "peaks": {"IDA": 0.8, "IDC": 0.8, "IDH": 0.8},
           "status": {"IDB": "missing_reading", "IDD": "conflicting_duplicate", "IDE": "negative_interval",
                      "IDF": "implausible_interval", "IDG": "missing_reading"},
           "offgrid": 1, "exact_dup": 1, "conflict_dup": 1}, open(sys.argv[2], "w"), indent=1)
print(len(rows), "synthetic rows")
