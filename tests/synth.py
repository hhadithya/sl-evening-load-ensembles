"""Synthetic fixtures (NOT real data) used by the regression tests. Importable helper module."""
import datetime as dt
import hashlib
import zipfile
from pathlib import Path

HEADER = ("household_ID,timestamp,date,time,obis,importkwh(kwh),exportkwh(kwh),importkvarh(kvarh),exportkvarh(kvarh),"
          "powerfactor,avgcurrent(v),avgexportkw(kw),avgimportkw(kw),avgvoltage(v),phaseainstcurrent(a),"
          "phaseainstvoltage(v),phasebinstcurrent(a),phasebinstvoltage(v),phasecinstcurrent(a),phasecinstvoltage(v)")
MEMBER = "data/consumption_data/smart_meter/15min_interval/smart_15min_{}.csv"


def ts(d, hh, mm):
    utc = dt.datetime.fromisoformat(f"{d}T{hh:02d}:{mm:02d}:00") - dt.timedelta(hours=5, minutes=30)
    ms = int(utc.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    return f"{float(round(ms, -7)):.1f}"


def row(h, d, hh, mm, imp, exp=0.0, ai=None, time_str=None, date_str=None):
    """One CSV row. time_str / date_str override the formatted strings (for invalid-value tests)."""
    ai = "" if ai is None else f"{ai}"
    t = time_str if time_str is not None else f"{hh:02d}:{mm:02d}:00"
    dd = date_str if date_str is not None else d
    return f"{h},{ts(d, hh, mm)},{dd},{t},LP,{imp},{exp},,,0.9,,0.0,{ai},230,,,,,,"


def day_rows(h, d, start, inc, skip=(), slots=range(96)):
    """Rows of one household-day. inc(s) = register increase over the interval ENDING at slot s."""
    out, R = [], start
    for s in range(96):
        if s > 0:
            R = round(R + inc(s), 6)
        if s in skip or s not in slots:
            continue
        hh, mm = divmod(s * 15, 60)
        out.append(row(h, d, hh, mm, R, ai=round(inc(s) * 4, 6)))
    return out


HEADER6 = ("household_ID,BRANCH,DEPOT,substation_ID,SUB_TYPE,E_BILLING,EXPORT,DATE,TIME,TOTAL_IMPORT (kWh),"
           "TOTAL_EXPORT (kWh),TOTAL_IMPORT - PV1 (kWh),TOTAL_EXPORT - PV1 (kWh),TR1_TOTAL_IMPORT (kWh),"
           "TR2_TOTAL_IMPORT (kWh),TR3_TOTAL_IMPORT (kWh),TR1_TOTAL_EXPORT (kWh),TR2_TOTAL_EXPORT (kWh),"
           "TR3_TOTAL_EXPORT (kWh),TR1_TOTAL_IMPORT - PV1 (kWh),TR2_TOTAL_IMPORT - PV1 (kWh),"
           "TR3_TOTAL_IMPORT - PV1 (kWh),TR1_TOTAL_EXPORT - PV1 (kWh),TR2_TOTAL_EXPORT - PV1 (kWh),"
           "TR3_TOTAL_EXPORT - PV1 (kWh),PHASE_A_CURRENT (A),PHASE_A_VOLTAGE (V),FREQUENCY (Hz),"
           "PHASE_B_CURRENT (A),PHASE_B_VOLTAGE (V),PHASE_C_CURRENT (A),PHASE_C_VOLTAGE (V)")
MEMBER6 = "data/consumption_data/smart_meter/6hour_interval/smart_6hour_{}.csv"


def row6(h, date, time, total_import, tr2):
    cols = HEADER6.split(",")
    d = {"household_ID": h, "DATE": date, "TIME": time, "TOTAL_IMPORT (kWh)": total_import,
         "TR2_TOTAL_IMPORT (kWh)": tr2, "BRANCH": "X", "DEPOT": "Y", "substation_ID": "s1"}
    return ",".join(str(d.get(c, "")) for c in cols)


def build_zip(path, f1=(), f2=(), f3=(), h6=()):
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, rows in zip((1, 2, 3), (f1, f2, f3)):
            z.writestr(MEMBER.format(k), HEADER + "\n" + "".join(r + "\n" for r in rows))
        if h6:
            z.writestr(MEMBER6.format(1), HEADER6 + "\n" + "".join(r + "\n" for r in h6))
    return path


def md5(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
