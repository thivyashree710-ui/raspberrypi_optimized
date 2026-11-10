"""Web dashboard for New_optimized.py (started with `--web`) - same look and page structure as the Jetson Xavier NX dashboard
(JETSON_XAVIER_NX_MIC710AIX repository): Live Monitor, Recordings, Reports (Hourly / Daily / Monthly lists, Custom Report Builder, Live Vehicle Report).
Runs in its OWN process; only the Python standard library + OpenCV/numpy are used, no internet is needed (stylesheet, logo and fonts are in web_assets/).
The detector copies each finished (annotated) picture into shared memory; this process streams it (MJPEG), records video, stores every counted vehicle in a small SQLite
database (so the reports survive restarts) and serves the pages.
There is NO login: anybody on the network who can reach the port can open the pages and start/stop/delete recordings. Use a trusted network, or --web-host 127.0.0.1."""
import os, re, io, json, time, threading, datetime, sqlite3, shutil, zipfile, calendar, html as _html
import numpy as np, cv2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode, quote

HERE = os.path.dirname(os.path.abspath(__file__)); ASSETS = os.path.join(HERE, "web_assets")
S = {"status": {}, "status_t": 0.0, "events": [], "jpg": None, "jpg_id": 0, "cond": threading.Condition(), "clients": 0,
     "rec": {"on": False, "file": None, "t0": 0.0, "writer": None, "frames": 0, "last_write": 0.0, "gaps": 0, "last_final": None, "last_final_t": None, "failed": []},
     "cfg": {}, "cmd_q": None, "stop": None, "lock": threading.Lock(), "dblock": threading.Lock(), "latest": None, "latest_id": 0, "latest_t": 0.0, "dur_cache": {}, "cpu_last": None, "started": time.time()}
SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$"); DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$"); TIME = re.compile(r"^\d{2}:\d{2}$"); LIGHT = {"two_wheeler", "cycle", "auto_rickshaw"}
esc = lambda t: _html.escape(str(t if t is not None else ""), quote=True)

# ---------------------------------------------------------------------------------------------------- icons + page shell (the Jetson navigation, topbar and status chips)
_SV = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">%s</svg>'
ICON_LIVE = _SV % '<rect x="2.5" y="6" width="14" height="12" rx="2"/><path d="M16.5 10.5 21 8v8l-4.5-2.5"/>'
ICON_REC = _SV % '<rect x="2.5" y="5" width="19" height="14" rx="2"/><line x1="7" y1="5" x2="7" y2="19"/><line x1="17" y1="5" x2="17" y2="19"/><line x1="2.5" y1="10" x2="7" y2="10"/><line x1="2.5" y1="14" x2="7" y2="14"/><line x1="17" y1="10" x2="21.5" y2="10"/><line x1="17" y1="14" x2="21.5" y2="14"/>'
ICON_REPORTS = _SV % '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/>'
ICON_CAM = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 8.5v7a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h4l1.5-2h7L17 7.5h4a1 1 0 0 1 1 1z"/><circle cx="12" cy="12.5" r="3.2"/></svg>'
ICON_ETH = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="4"/><path d="M7 8h10l-1.3 7h-7.4z"/><line x1="8.2" y1="8" x2="8.2" y2="10.2"/><line x1="10" y1="8" x2="10" y2="10.2"/><line x1="11.8" y1="8" x2="11.8" y2="10.2"/><line x1="13.6" y1="8" x2="13.6" y2="10.2"/><line x1="15.4" y1="8" x2="15.4" y2="10.2"/></svg>'

NAV_JS = """<script>
async function _navRefreshLiveStatus() {
  try {
    const ops = await (await fetch('/api/operations')).json();
    const camOnline = ops.camera.status === 'ONLINE';
    document.getElementById('status-camera').classList.toggle('status-bad', !camOnline); document.getElementById('status-camera-val').textContent = ops.camera.status;
    document.getElementById('status-ethernet').classList.toggle('status-bad', ops.network.ethernet_status === 'OFFLINE'); document.getElementById('status-ethernet-val').textContent = ops.network.ethernet_status;
    const statusEl = document.getElementById('topbar-status'); const healthColors = {HEALTHY: '#22a06b', WARNING: '#e0982a', CRITICAL: '#e0555a'}; const healthText = {HEALTHY: 'SYSTEM ONLINE', WARNING: 'SYSTEM WARNING', CRITICAL: 'SYSTEM CRITICAL'};
    const h = ops.system_health.status; statusEl.style.color = healthColors[h] || '#22a06b'; statusEl.innerHTML = '&#9679; ' + (healthText[h] || 'SYSTEM ONLINE');
    document.getElementById('health-bar').style.background = healthColors[h] || '#22a06b';
  } catch (e) { console.error('nav live status refresh failed', e); }
}
_navRefreshLiveStatus(); setInterval(_navRefreshLiveStatus, 5000);
</script>"""


def nav(active, title, sub, ops):
    items = [("/", "live", "Live Monitor", ICON_LIVE), ("/recordings", "recordings", "Recordings", ICON_REC), ("/custom", "reports", "Reports", ICON_REPORTS)]
    rows = "".join('<a href="%s" class="nav-item%s"><span class="nav-icon">%s</span><span class="nav-label">%s</span></a>' % (h, " active" if k == active else "", ic, tx) for h, k, tx, ic in items)
    h = ops["system_health"]["status"]; hc = {"HEALTHY": "var(--good)", "WARNING": "var(--warn)", "CRITICAL": "var(--bad)"}[h]; st = {"HEALTHY": "SYSTEM ONLINE", "WARNING": "SYSTEM WARNING", "CRITICAL": "SYSTEM CRITICAL"}[h]
    cam_ok = ops["camera"]["status"] == "ONLINE"; eth_bad = ops["network"]["ethernet_status"] == "OFFLINE"
    return ('<div id="health-bar" style="background:%s"></div><div class="app"><div class="sidebar" id="sidebar"><div class="sidebar-brand"><div class="logo-row"><img src="/static/logo.png" class="logo-img" alt="Company logo"></div>'
            '<div class="logo-text">ITS CONTROL</div><div class="logo-sub">Traffic Intelligence</div></div><div class="nav-group">%s</div></div><div class="main"><div class="topbar"><div class="topbar-left"><h1 class="page-title">%s</h1>%s</div>'
            '<div class="topbar-right"><span class="hero-mini-status%s" id="status-camera">%s<span id="status-camera-val">%s</span></span><span class="hero-mini-status%s" id="status-ethernet">%s<span id="status-ethernet-val">%s</span></span>'
            '<div class="topbar-status" id="topbar-status" style="color:%s">&#9679; %s</div></div></div>%s') % (
        hc, rows, esc(title), ('<div class="page-sub">%s</div>' % esc(sub)) if sub else "", "" if cam_ok else " status-bad", ICON_CAM, ops["camera"]["status"], " status-bad" if eth_bad else "", ICON_ETH, ops["network"]["ethernet_status"], hc, st, NAV_JS)


def shell(active, title, sub, body, script=""):
    ops = api_operations()
    return ('<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>%s - ITS Control</title><link rel="stylesheet" href="/static/style.css"></head><body>%s<div class="wrap">%s</div></div></div>%s</body></html>'
            % (esc(title), nav(active, title, sub, ops), body, script))


# ---------------------------------------------------------------------------------------------------- data: database, operations, statistics
def dbq(sql, args=()):
    with S["dblock"]:
        con = sqlite3.connect(S["db"], timeout=10)
        try: cur = con.execute(sql, args); rows = cur.fetchall(); con.commit(); return rows, cur.lastrowid
        finally: con.close()


def db_init():
    os.makedirs(S["cfg"]["data_dir"], exist_ok=True); S["db"] = os.path.join(S["cfg"]["data_dir"], "traffic.db")
    dbq("CREATE TABLE IF NOT EXISTS crossings (id INTEGER PRIMARY KEY, ts REAL, day TEXT, hour INTEGER, cls TEXT, dir TEXT, track INTEGER, frame INTEGER, conf REAL, snap TEXT)"); dbq("CREATE INDEX IF NOT EXISTS ix_day ON crossings(day)"); dbq("CREATE INDEX IF NOT EXISTS ix_ts ON crossings(ts)")


def add_event(ev):
    now = datetime.datetime.now(); day = now.strftime("%Y-%m-%d"); conf = ev.get("conf")
    _, rid = dbq("INSERT INTO crossings (ts, day, hour, cls, dir, track, frame, conf, snap) VALUES (?,?,?,?,?,?,?,?,NULL)", (now.timestamp(), day, now.hour, ev["class"], ev["dir"], ev["id"], ev["frame"], conf))
    if ev.get("jpg"):
        d = os.path.join(S["cfg"]["data_dir"], "snaps", day); os.makedirs(d, exist_ok=True); p = os.path.join(d, "%d.jpg" % rid)
        with open(p, "wb") as fh: fh.write(ev["jpg"])
        dbq("UPDATE crossings SET snap=? WHERE id=?", (p, rid))
    S["events"].append(now.timestamp()); del S["events"][:-3000]


def cpu_percent():
    try:
        with open("/proc/stat") as fh: p = fh.readline().split()[1:]
        v = [int(x) for x in p[:8]]; idle = v[3] + v[4]; tot = sum(v); last = S["cpu_last"]; S["cpu_last"] = (tot, idle)
        if last and tot > last[0]: return round(100.0 * (1 - (idle - last[1]) / (tot - last[0])), 1)
    except Exception: pass
    return None


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh: return round(int(fh.read().strip()) / 1000.0, 1)
    except Exception: return None


def eth_status():
    try:
        st = []
        for n in os.listdir("/sys/class/net"):
            if n.startswith(("eth", "en")):
                with open("/sys/class/net/%s/operstate" % n) as fh: st.append(fh.read().strip())
        return "ONLINE" if "up" in st else ("OFFLINE" if st else "UNKNOWN")
    except Exception: return "UNKNOWN"


def disk_info():
    try: u = shutil.disk_usage(S["cfg"]["rec_dir"]); return {"used_pct": round(100.0 * u.used / u.total, 1), "free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)}
    except OSError: return None


def api_operations():
    st = S["status"]; age = time.time() - S["status_t"]; cam_on = bool(st) and age < 6.0; r = S["rec"]; d = disk_info(); temp = cpu_temp(); pen = 0
    health = "HEALTHY"
    if not cam_on: health = "CRITICAL"; pen += 60
    if temp is not None and temp >= 80: health = "WARNING" if health == "HEALTHY" else health; pen += 15
    if d and d["used_pct"] >= 90: health = "WARNING" if health == "HEALTHY" else health; pen += 15
    now = time.time(); vpm = round(sum(1 for t in S["events"] if now - t <= 300) / 5.0, 1); lvl = "LOW" if vpm < 5 else "MODERATE" if vpm < 15 else "HEAVY" if vpm < 30 else "CONGESTED"
    return {"camera": {"status": "ONLINE" if cam_on else "OFFLINE", "stream": st.get("stream")}, "network": {"ethernet_status": eth_status()},
            "ai_engine": {"effective_fps": st.get("fps") if cam_on else None, "fps": st.get("detect") if cam_on else None, "infer_ms": st.get("infer_ms"), "mode": st.get("mode")},
            "hardware": {"cpu_pct": cpu_percent(), "temp_c": temp}, "recording": {"status": "ACTIVE" if r["on"] else "STOPPED", "duration_seconds": int(now - r["t0"]) if r["on"] else None, "filename": r["file"] if r["on"] else None},
            "storage": {"used_pct": d["used_pct"] if d else None, "free_gb": d["free_gb"] if d else None}, "system_health": {"status": health, "percent": max(0, 100 - pen)}, "traffic": {"vehicles_per_minute": vpm, "level": lvl}}


def _vehicle(row):
    rid, day, ts, cls, d, conf, snap = row; t = datetime.datetime.fromtimestamp(ts)
    return {"id": rid, "date": day, "time": t.strftime("%H:%M:%S"), "type": cls, "direction": d, "confidence": round(conf * 100, 2) if conf is not None else None, "snapshot": ("%d.jpg" % rid) if snap else None}


def api_stats():
    day = datetime.date.today().isoformat(); rows, _ = dbq("SELECT hour, dir, COUNT(*) FROM crossings WHERE day=? GROUP BY hour, dir", (day,)); by_hour = [0] * 24; up = down = 0
    for h, d, n in rows: by_hour[h] += n; up += n if d == "IN" else 0; down += n if d == "OUT" else 0
    rec, _ = dbq("SELECT id, day, ts, cls, dir, conf, snap FROM crossings WHERE day=? ORDER BY id DESC LIMIT 40", (day,)); y = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    yt = dbq("SELECT COUNT(*) FROM crossings WHERE day=?", (y,))[0][0][0]; tot = up + down; peak = max(range(24), key=lambda i: by_hour[i]) if tot else None
    bt, _ = dbq("SELECT cls, COUNT(*) FROM crossings WHERE day=? GROUP BY cls ORDER BY COUNT(*) DESC", (day,))
    return {"total": tot, "up": up, "down": down, "by_hour": by_hour, "recent": [_vehicle(r) for r in rec], "by_type": [{"type": c, "count": n} for c, n in bt], "peak_hour": peak, "peak_hour_count": by_hour[peak] if peak is not None else 0,
            "vs_yesterday_pct": round(100.0 * (tot - yt) / yt, 1) if yt else None, "camera_connected": api_operations()["camera"]["status"] == "ONLINE"}


def api_intel():
    day = datetime.date.today().isoformat(); rows, _ = dbq("SELECT ts, hour, cls FROM crossings WHERE day=? ORDER BY ts", (day,)); n = len(rows)
    out = {"avg_vehicles_per_minute": None, "busiest_15min_start": None, "busiest_15min_count": 0, "hour_vs_previous_pct": None, "current_hour_count": 0, "previous_hour_count": 0, "avg_gap_seconds": None, "class_breakdown": []}
    now_h = datetime.datetime.now().hour; out["current_hour_count"] = sum(1 for r in rows if r[1] == now_h); out["previous_hour_count"] = sum(1 for r in rows if r[1] == now_h - 1)
    if out["previous_hour_count"]: out["hour_vs_previous_pct"] = round(100.0 * (out["current_hour_count"] - out["previous_hour_count"]) / out["previous_hour_count"], 1)
    if n >= 2:
        span = max(1.0, (rows[-1][0] - rows[0][0]) / 60.0); out["avg_vehicles_per_minute"] = round(n / span, 2); out["avg_gap_seconds"] = round((rows[-1][0] - rows[0][0]) / (n - 1), 1)
    if n:
        b = {}
        for ts, h, c in rows: t = datetime.datetime.fromtimestamp(ts); k = (t.hour, t.minute // 15); b[k] = b.get(k, 0) + 1
        k = max(b, key=b.get); out["busiest_15min_start"] = "%02d:%02d" % (k[0], k[1] * 15); out["busiest_15min_count"] = b[k]; cc = {}
        for _, _, c in rows: cc[c] = cc.get(c, 0) + 1
        out["class_breakdown"] = [{"type": c, "pct": round(100.0 * v / n, 1)} for c, v in sorted(cc.items(), key=lambda x: -x[1])]
    return out


def rec_start_time(name):
    m = re.match(r"rec_(\d{8})_(\d{6})", name)
    if m:
        try: return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError: pass
    return None


def video_seconds(path, mtime):
    k = (path, mtime)
    if k not in S["dur_cache"]:
        try: c = cv2.VideoCapture(path); n = c.get(cv2.CAP_PROP_FRAME_COUNT); f = c.get(cv2.CAP_PROP_FPS) or 25.0; c.release(); S["dur_cache"][k] = (n / f) if n and n > 0 else None
        except Exception: S["dur_cache"][k] = None
    return S["dur_cache"][k]


def api_recordings(q):
    d = S["cfg"]["rec_dir"]; out = []; r = S["rec"]; view = (q.get("view") or [""])[0]; date = (q.get("date") or [""])[0]; month = (q.get("month") or [""])[0]
    try: names = sorted(os.listdir(d), reverse=True)
    except OSError: names = []
    for n in names:
        p = os.path.join(d, n)
        if not (n.lower().endswith((".mp4", ".webm")) and os.path.isfile(p)): continue
        st = os.stat(p); t = rec_start_time(n) or datetime.datetime.fromtimestamp(st.st_mtime); ds = t.strftime("%Y-%m-%d")
        if view == "hourly" and date and (ds != date or t.hour != int((q.get("hour") or ["-1"])[0] or -1)): continue
        if view == "custom":
            try:
                a = datetime.datetime.strptime("%s %s" % (q["from_date"][0], (q.get("from_time") or ["00:00"])[0]), "%Y-%m-%d %H:%M"); b = datetime.datetime.strptime("%s %s" % (q["to_date"][0], (q.get("to_time") or ["23:59"])[0]), "%Y-%m-%d %H:%M")
                if not (a <= t <= b + datetime.timedelta(seconds=59)): continue
            except (KeyError, ValueError): pass
        elif view != "hourly" and date and ds != date: continue
        if month and not ds.startswith(month): continue
        rec = bool(r["on"] and r["file"] == n)
        out.append({"filename": n, "date": ds, "time": t.strftime("%H:%M:%S"), "size_mb": round(st.st_size / 1e6, 1), "status": "RECORDING" if rec else "READY", "duration": (time.time() - r["t0"]) if rec else video_seconds(p, st.st_mtime)})
    return {"recordings": out[:500], "total": len(out), "is_recording": r["on"], "current_filename": r["file"] if r["on"] else None, "current_mode": "manual"}


def api_rec_health():
    r = S["rec"]; size = None
    if r["on"]:
        try: size = os.path.getsize(os.path.join(S["cfg"]["rec_dir"], r["file"]))
        except OSError: pass
    st = "HEALTHY"
    if r["on"] and time.time() - r["last_write"] > 3.0: st = "WARNING"
    if r["failed"] and not r["on"] and r["failed"][-1].get("recent"): st = "WARNING"
    return {"status": st, "current_file": r["file"] if r["on"] else None, "current_file_size_bytes": size, "recording_fps": round(r["frames"] / max(time.time() - r["t0"], 1e-3), 1) if r["on"] else None,
            "last_frame_written_at": int(r["last_write"] * 1000) if r["last_write"] else None, "last_finalized_file": r["last_final"], "last_finalized_time": int(r["last_final_t"] * 1000) if r["last_final_t"] else None,
            "detected_gaps": [0] * r["gaps"], "failed_recordings": r["failed"]}


def start_rec():
    with S["lock"]:
        r = S["rec"]
        if r["on"] or S["latest"] is None: return False
        os.makedirs(S["cfg"]["rec_dir"], exist_ok=True); h, w = S["latest"].shape[:2]; name = "rec_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"; path = os.path.join(S["cfg"]["rec_dir"], name); wr = None
        for fc in ("avc1", "mp4v"):                                             # H.264 plays in every browser; mp4v is the fallback
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), 25.0, (w, h))
            if wr.isOpened(): break
            wr.release(); wr = None
        if wr is None: r["failed"].append({"time": time.time(), "file": name, "recent": True}); return False
        r.update(on=True, file=name, t0=time.time(), writer=wr, frames=0, last_write=time.time(), gaps=0); return True


def stop_rec():
    with S["lock"]:
        r = S["rec"]
        if not r["on"]: return
        r["on"] = False; r["last_final"] = r["file"]; r["last_final_t"] = time.time()
        if r["writer"] is not None: r["writer"].release()
        r["writer"] = None


# ---------------------------------------------------------------------------------------------------- reports: filtered vehicle list, Excel files
def _ts(d, t): return datetime.datetime.strptime("%s %s" % (d, t), "%Y-%m-%d %H:%M").timestamp()


def vehicle_query(q, limit=None, offset=0, sort="timestamp", sort_dir="desc"):
    g = lambda k, d="": (q.get(k) or [d])[0]; where, params = [], []
    if g("search").strip(): where.append("CAST(id AS TEXT) LIKE ?"); params.append("%" + g("search").strip() + "%")
    vt = q.get("vtype", [])
    if vt: where.append("cls IN (%s)" % ",".join("?" for _ in vt)); params += vt
    if g("direction", "all") in ("IN", "OUT"): where.append("dir = ?"); params.append(g("direction"))
    sd, ed = g("start_date"), g("end_date")
    try:
        if DATE.match(sd): where.append("ts >= ?"); params.append(_ts(sd, g("start_time") if TIME.match(g("start_time")) else "00:00"))
        if DATE.match(ed): where.append("ts <= ?"); params.append(_ts(ed, g("end_time") if TIME.match(g("end_time")) else "23:59") + 59)
    except ValueError: pass
    w = ("WHERE " + " AND ".join(where)) if where else ""; col = {"id": "id", "timestamp": "ts", "vehicle_type": "cls", "direction": "dir", "confidence": "conf"}.get(sort, "ts"); sort_dir = "ASC" if sort_dir == "asc" else "DESC"
    total = dbq("SELECT COUNT(*) FROM crossings " + w, params)[0][0][0]; sql = "SELECT id, day, ts, cls, dir, conf, snap FROM crossings %s ORDER BY %s %s, id DESC" % (w, col, sort_dir)
    if limit: sql += " LIMIT ? OFFSET ?"; params = params + [limit, offset]
    return total, dbq(sql, params)[0]


def _col(n):
    s = ""
    while n: n, r = divmod(n - 1, 26); s = chr(65 + r) + s
    return s


def make_xlsx(sheets):
    """A minimal .xlsx (Office Open XML) written with the standard library only. sheets = [(name, header, rows)]."""
    def cell(ref, v, bold=False):
        s = ' s="1"' if bold else ""
        if isinstance(v, (int, float)): return '<c r="%s"%s><v>%s</v></c>' % (ref, s, v)
        return '<c r="%s"%s t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>' % (ref, s, _html.escape(str(v)))
    bio = io.BytesIO(); z = zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED)
    z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + "".join('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (i + 1) for i in range(len(sheets))) + "</Types>")
    z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    z.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (_html.escape(n[:31]), i + 1, i + 1) for i, (n, _, _) in enumerate(sheets)) + "</sheets></workbook>")
    z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i + 1, i + 1) for i in range(len(sheets))) + '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>' % (len(sheets) + 1))
    z.writestr("xl/styles.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
    for i, (name, header, rows) in enumerate(sheets):
        out = ['<row r="1">' + "".join(cell("%s1" % _col(c + 1), h, True) for c, h in enumerate(header)) + "</row>"]
        for r, row in enumerate(rows, 2): out.append('<row r="%d">' % r + "".join(cell("%s%d" % (_col(c + 1), r), v) for c, v in enumerate(row)) + "</row>")
        z.writestr("xl/worksheets/sheet%d.xml" % (i + 1), '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cols><col min="1" max="%d" width="18" customWidth="1"/></cols><sheetData>%s</sheetData></worksheet>' % (len(header), "".join(out)))
    z.close(); return bio.getvalue()


def report_xlsx(start_ts, end_ts, vtype="all", direction="all", title="Report"):
    where, params = ["ts >= ?", "ts <= ?"], [start_ts, end_ts]
    if vtype != "all": where.append("cls = ?"); params.append(vtype)
    if direction in ("IN", "OUT"): where.append("dir = ?"); params.append(direction)
    rows, _ = dbq("SELECT id, day, ts, cls, dir, conf FROM crossings WHERE %s ORDER BY ts" % " AND ".join(where), params); vs = []; summ = {}
    for rid, day, ts, cls, d, conf in rows:
        vs.append((rid, day, datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S"), cls, d, round(conf * 100, 2) if conf is not None else "")); e = summ.setdefault(cls, [0, 0]); e[0 if d == "IN" else 1] += 1
    sm = [(c, v[0], v[1], v[0] + v[1]) for c, v in sorted(summ.items(), key=lambda x: -(x[1][0] + x[1][1]))] + [("TOTAL", sum(v[0] for v in summ.values()), sum(v[1] for v in summ.values()), len(vs))]
    return make_xlsx([("Vehicles", ["ID", "Date", "Time", "Vehicle class", "Direction", "Confidence %"], vs), ("Summary", ["Vehicle class", "IN", "OUT", "Total"], sm)]), len(vs)


def report_files(kind):
    if kind == "hourly": rows, _ = dbq("SELECT DISTINCT day, hour FROM crossings ORDER BY day DESC, hour DESC"); return ["hourly_%s_%02d00.xlsx" % (d, h) for d, h in rows]
    if kind == "daily": rows, _ = dbq("SELECT DISTINCT day FROM crossings ORDER BY day DESC"); return ["daily_%s.xlsx" % d for (d,) in rows]
    rows, _ = dbq("SELECT DISTINCT substr(day,1,7) FROM crossings ORDER BY 1 DESC"); return ["monthly_%s.xlsx" % m for (m,) in rows]


def report_range(name):
    m = re.match(r"^hourly_(\d{4}-\d{2}-\d{2})_(\d{2})00\.xlsx$", name)
    if m: d, h = m.group(1), int(m.group(2)); return _ts(d, "%02d:00" % h), _ts(d, "%02d:59" % h) + 59, (d, "%02d:00" % h, d, "%02d:59" % h)
    m = re.match(r"^daily_(\d{4}-\d{2}-\d{2})\.xlsx$", name)
    if m: d = m.group(1); return _ts(d, "00:00"), _ts(d, "23:59") + 59, (d, "00:00", d, "23:59")
    m = re.match(r"^monthly_(\d{4})-(\d{2})\.xlsx$", name)
    if m: y, mo = int(m.group(1)), int(m.group(2)); last = calendar.monthrange(y, mo)[1]; a, b = "%04d-%02d-01" % (y, mo), "%04d-%02d-%02d" % (y, mo, last); return _ts(a, "00:00"), _ts(b, "23:59") + 59, (a, "00:00", b, "23:59")
    return None


# ---------------------------------------------------------------------------------------------------- pages
LIVE_BODY = r"""
<style>
.live-mon-grid { grid-template-columns: 1fr 1.35fr !important; }
.live-cam-card, .live-recent-card { position: relative; overflow: hidden; }
.live-cam-card::before, .live-recent-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.live-cam-card::before { background: linear-gradient(90deg, #f87171, #fb7185); } .live-recent-card::before { background: linear-gradient(90deg, #818cf8, #6366f1); }
.lv-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
.lv-viewall { color: #5b61e8; font-size: 12px; font-weight: 800; text-decoration: none; letter-spacing: 0.3px; white-space: nowrap; } .lv-viewall:hover { text-decoration: underline; }
.lv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(98px, 1fr)); gap: 10px; align-content: start; padding: 2px 6px 4px 2px; }
.lv-empty { grid-column: 1 / -1; color: var(--muted); padding: 16px; text-align: center; font-size: 13px; }
.lv-tile { --tile-c: #14b8a6; position: relative; background: var(--panel-raised); border: 1px solid var(--border-soft); border-radius: 14px; padding: 12px 7px 8px; display: flex; flex-direction: column; gap: 6px; min-width: 0; cursor: pointer; box-shadow: var(--shadow-soft); transition: transform 0.15s ease, box-shadow 0.15s ease; overflow: hidden; }
.lv-tile::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--tile-c); }
.lv-tile:hover { transform: translateY(-2px); box-shadow: var(--shadow-lift); }
.lv-tile:nth-child(10n+1) { --tile-c: #14b8a6; } .lv-tile:nth-child(10n+2) { --tile-c: #f59e0b; } .lv-tile:nth-child(10n+3) { --tile-c: #ec4899; } .lv-tile:nth-child(10n+4) { --tile-c: #eab308; } .lv-tile:nth-child(10n+5) { --tile-c: #22c55e; }
.lv-tile:nth-child(10n+6) { --tile-c: #ef4444; } .lv-tile:nth-child(10n+7) { --tile-c: #8b5cf6; } .lv-tile:nth-child(10n+8) { --tile-c: #3b82f6; } .lv-tile:nth-child(10n+9) { --tile-c: #f43f5e; } .lv-tile:nth-child(10n+10) { --tile-c: #a855f7; }
.lv-top { display: flex; align-items: center; gap: 4px; min-width: 0; font-size: 11px; }
.lv-id { flex: none; background: #e7e4ff; color: #5b4fe0; font-weight: 800; border-radius: 6px; padding: 1px 6px; font-family: var(--font-mono); font-size: 10.5px; }
.lv-type { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; color: var(--text); } .lv-type::before { content: '\2022'; margin-right: 4px; color: var(--ink-dim); }
.lv-img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px; background: #2a2d3a; display: block; }
.lv-noimg { width: 100%; aspect-ratio: 1 / 1; border-radius: 8px; background: var(--surface-2); color: var(--muted); display: flex; align-items: center; justify-content: center; font-size: 11px; }
.lv-time { text-align: center; font-weight: 800; font-size: 11.5px; white-space: nowrap; font-variant-numeric: tabular-nums; color: var(--text); } .lv-time span { color: var(--ink-dim); font-weight: 700; }
.lv-badges { display: flex; justify-content: center; gap: 4px; flex-wrap: wrap; } .lv-b { display: inline-flex; align-items: center; gap: 2px; font-size: 8.5px; font-weight: 800; letter-spacing: 0.2px; padding: 2px 6px; border-radius: 20px; white-space: nowrap; }
.lv-b.ok { background: rgba(47,158,79,0.13); color: var(--good); } .lv-b.dn { background: rgba(224,85,106,0.13); color: #e0556a; }
</style>
<div id="camera-banner" style="display:none; background:rgba(239,68,68,0.1); border:1px solid var(--bad); color:var(--bad); padding:14px 18px; border-radius:10px; margin-bottom:16px; font-weight:600;"></div>
<div id="connection-lost-banner" style="display:none; background:var(--bad); border:1px solid var(--bad); color:#fff; padding:16px 18px; border-radius:10px; margin-bottom:16px; font-weight:700; font-size:14.5px; box-shadow:0 4px 16px -4px rgba(239,68,68,0.5);"></div>
<div class="card" id="widget-section-ops" style="margin-bottom:16px; padding:16px 18px;">
<h2 style="font-size:11px; margin-bottom:8px;"><span class="live-dot"></span>System Operations</h2>
<div class="kpi-grid" id="ops-rows" style="margin-bottom:0;">
  <div class="kpi-card"><div class="kpi-label">AI Engine</div><div class="kpi-value" style="font-size:20px;" id="ops-ai-fps">--</div></div>
  <div class="kpi-card"><div class="kpi-label">CPU / Temperature</div><div class="kpi-value" style="font-size:20px;" id="ops-cpu-gpu-pct">--</div></div>
  <div class="kpi-card"><div class="kpi-label">Recording</div><div class="kpi-value" style="font-size:20px;" id="ops-recording-mode">--</div></div>
  <div class="kpi-card"><div class="kpi-label">Storage</div><div class="kpi-value" style="font-size:20px;" id="ops-storage-pct">--</div></div>
</div></div>
<div class="grid live-mon-grid" style="margin-top:28px;">
<div class="card live-cam-card">
<div style="display:flex; align-items:baseline; justify-content:space-between; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
  <div style="display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;"><h2 style="text-transform:none; letter-spacing:0; margin:0;">Live Monitoring</h2><span id="src-info" style="font-size:11px; color:var(--ink-dim); font-weight:700;"></span></div>
  <div style="display:flex; align-items:center; gap:10px;"><span><span class="live-dot"></span><b style="font-size:11px; letter-spacing:0.5px; color:var(--text);">LIVE</b></span><a href="#" class="topbar-icon-btn" id="fullscreen-btn" title="Full-screen Control Room">&#x26F6;</a></div>
</div>
<div class="hero-cam hero-cam-fill" id="hero-cam"><img src="/stream" class="stream-img" alt="Live annotated camera feed" id="stream-img">
  <div class="cam-offline-overlay" id="cam-offline-overlay"><div class="spinner"></div><div class="cam-offline-title">CAMERA OFFLINE</div><div class="cam-offline-sub">Attempting to reconnect&hellip;</div></div></div>
</div>
<div class="card live-recent-card" id="widget-section-recent"><div class="lv-head"><h2 style="margin:0;">Latest Vehicles (Counted)</h2><a class="lv-viewall" href="/custom">VIEW ALL &rarr;</a></div>
<div class="vehicle-row-list-wrap"><div class="lv-grid" id="recent-list"><div class="lv-empty">Loading&hellip;</div></div></div></div>
</div>
<div class="kpi-grid" style="margin-top:16px;">
  <div class="kpi-card"><div class="kpi-label">Total Today</div><div class="kpi-value" id="stat-total">--</div><div class="kpi-context" id="stat-vs-yest">&nbsp;</div></div>
  <div class="kpi-card"><div class="kpi-label">In</div><div class="kpi-value" id="stat-up">--</div><div class="kpi-context" id="stat-up-pct">&nbsp;</div></div>
  <div class="kpi-card"><div class="kpi-label">Out</div><div class="kpi-value" id="stat-down">--</div><div class="kpi-context" id="stat-down-pct">&nbsp;</div></div>
  <div class="kpi-card"><div class="kpi-label">Peak Hour</div><div class="kpi-value" style="font-size:20px;" id="stat-peak-hour">&mdash;</div><div class="kpi-context" id="stat-peak-hour-count">No data yet</div></div>
</div>
<button class="cr-exit-btn" id="cr-exit-btn">&#10005; Exit Control Room</button>
<div class="control-room-strip" id="control-room-strip">
  <div class="cr-metric"><div class="cr-label">Total</div><div class="cr-value" id="cr-total">--</div></div><div class="cr-metric"><div class="cr-label">In</div><div class="cr-value" id="cr-up">--</div></div>
  <div class="cr-metric"><div class="cr-label">Out</div><div class="cr-value" id="cr-down">--</div></div><div class="cr-metric"><div class="cr-label">Video FPS</div><div class="cr-value" id="cr-cam-fps">--</div></div>
  <div class="cr-metric"><div class="cr-label">Detections/s</div><div class="cr-value" id="cr-inf-fps">--</div></div><div class="cr-metric"><div class="cr-label">Veh/Min</div><div class="cr-value" id="cr-vpm">--</div></div>
  <div class="cr-metric"><div class="cr-label">Traffic</div><div class="cr-value" id="cr-traffic">--</div></div><div class="cr-metric"><div class="cr-label">Recording</div><div class="cr-value" id="cr-recording">--</div></div>
  <div class="cr-metric"><div class="cr-label">Camera</div><div class="cr-value" id="cr-camera">--</div></div><div class="cr-metric"><div class="cr-label">System Health</div><div class="cr-value" id="cr-health">--</div></div>
</div>
<div class="card" style="margin-bottom:16px;" id="widget-section-hourlychart"><h2><span class="live-dot"></span>Today's Traffic by Hour</h2><div class="bar-wrap" id="chart"></div><div class="bar-hours" id="chart-labels"></div></div>
<div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
  <div class="card ti-card ti-card-accent-1"><div class="ti-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="6" width="14" height="12" rx="2"/><path d="M16.5 10.5 21 8v8l-4.5-2.5"/></svg></div><div class="ti-label">Avg Vehicles/Min</div><div class="ti-value" id="ti-avg-vpm">--</div></div>
  <div class="card ti-card ti-card-accent-2"><div class="ti-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polyline points="12,7 12,12 15.5,14"/></svg></div><div class="ti-label">Busiest 15-Min Period</div><div class="ti-value" id="ti-busiest">--</div></div>
  <div class="card ti-card ti-card-accent-3"><div class="ti-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3,17 9,11 13,15 21,6"/><polyline points="15,6 21,6 21,12"/></svg></div><div class="ti-label">This Hour vs Last Hour</div><div class="ti-value" id="ti-hour-cmp">--</div></div>
</div>
<div class="grid" style="grid-template-columns: 1fr 1.4fr; margin-bottom:16px;">
<div class="card"><h2>Avg Gap Between Vehicles</h2><div class="ti-value" id="ti-avg-gap" style="font-size:32px; margin:8px 0;">--</div><div class="ti-ratio-labels"><span>Based on today's real vehicle timestamps</span></div></div>
<div class="card"><h2>Vehicle-Class Breakdown</h2><div id="ti-class-breakdown" class="chart-empty">Loading&hellip;</div></div>
</div>
<div class="snap-modal-backdrop" id="snap-modal-backdrop"><div class="snap-modal" id="snap-modal"><button class="snap-modal-close" id="snap-modal-close" title="Close" aria-label="Close">&#10005;</button>
  <div class="snap-modal-img-wrap" id="snap-modal-img-wrap"><img id="snap-modal-img" alt="Vehicle snapshot" onerror="handleModalImgError()"></div>
  <div class="snap-modal-details">
    <div class="snap-modal-row"><span class="snap-modal-label">Vehicle ID</span><span class="snap-modal-val" id="snap-modal-id">--</span></div><div class="snap-modal-row"><span class="snap-modal-label">Type</span><span class="snap-modal-val" id="snap-modal-type">--</span></div>
    <div class="snap-modal-row"><span class="snap-modal-label">Direction</span><span class="snap-modal-val" id="snap-modal-direction">--</span></div><div class="snap-modal-row"><span class="snap-modal-label">Date &amp; Time</span><span class="snap-modal-val" id="snap-modal-datetime">--</span></div>
    <div class="snap-modal-row"><span class="snap-modal-label">Confidence</span><span class="snap-modal-val" id="snap-modal-confidence">--</span></div><a href="#" target="_blank" class="dl-btn" id="snap-modal-original" style="margin-top:10px; text-align:center;">Open Original</a>
  </div></div></div>
"""
LIVE_JS = r"""<script>
function handleThumbError(imgEl) { const span = document.createElement('span'); span.className = 'thumb-unavailable'; span.textContent = 'No image'; imgEl.replaceWith(span); }
function handleModalImgError() { document.getElementById('snap-modal-img-wrap').innerHTML = '<div class="snap-unavailable-box">Snapshot unavailable</div>'; }
function openSnapshotModal(d) {
  const wrap = document.getElementById('snap-modal-img-wrap'); wrap.innerHTML = '<img id="snap-modal-img" alt="Vehicle snapshot" onerror="handleModalImgError()">'; document.getElementById('snap-modal-img').src = '/snapshot/' + d.snapshot;
  document.getElementById('snap-modal-id').textContent = '#' + d.id; document.getElementById('snap-modal-type').textContent = d.type; document.getElementById('snap-modal-direction').textContent = d.direction;
  document.getElementById('snap-modal-datetime').textContent = (d.date || '') + ' ' + (d.time || ''); document.getElementById('snap-modal-confidence').textContent = d.confidence ? d.confidence + '%' : '--';
  document.getElementById('snap-modal-original').href = '/snapshot/' + d.snapshot; document.getElementById('snap-modal-backdrop').classList.add('show');
}
function closeSnapshotModal() { document.getElementById('snap-modal-backdrop').classList.remove('show'); }
document.body.addEventListener('click', (e) => { const thumb = e.target.closest('.clickable-thumb') || (e.target.closest('.lv-tile') && e.target.closest('.lv-tile').querySelector('.clickable-thumb')); if (thumb) { openSnapshotModal(thumb.dataset); return; } if (e.target.id === 'snap-modal-backdrop') closeSnapshotModal(); });
document.getElementById('snap-modal-close')?.addEventListener('click', closeSnapshotModal);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSnapshotModal(); });
function enterControlRoom() { document.body.classList.add('control-room-mode'); document.documentElement.requestFullscreen().catch(() => {}); }
function exitControlRoom() { document.body.classList.remove('control-room-mode'); if (document.fullscreenElement) document.exitFullscreen(); }
document.getElementById('fullscreen-btn')?.addEventListener('click', (e) => { e.preventDefault(); if (!document.body.classList.contains('control-room-mode')) enterControlRoom(); else exitControlRoom(); });
document.getElementById('cr-exit-btn')?.addEventListener('click', exitControlRoom);
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement) document.body.classList.remove('control-room-mode'); });
function renderChart(byHour) {
  const max = Math.max(1, ...byHour); const chart = document.getElementById('chart'); const labels = document.getElementById('chart-labels'); chart.innerHTML = ''; labels.innerHTML = '';
  byHour.forEach((v, h) => {
    const bar = document.createElement('div'); bar.className = 'bar'; bar.style.height = Math.max(2, (v / max) * 110) + 'px'; const ratio = v / max;
    if (ratio <= 0.33) bar.style.background = 'linear-gradient(180deg, #fde68a, #f6c453)'; else if (ratio <= 0.66) bar.style.background = 'linear-gradient(180deg, #34d399, #10b981)'; else bar.style.background = 'linear-gradient(180deg, #fca5a5, #f0787a)';
    bar.title = h + ':00 - ' + v + ' vehicles'; chart.appendChild(bar); const lbl = document.createElement('span'); lbl.textContent = (h % 3 === 0) ? h : ''; labels.appendChild(lbl);
  });
}
function animateNum(el, newVal) { if (!el || el.textContent == newVal) return; el.textContent = newVal; el.style.transform = 'scale(1.1)'; setTimeout(() => el.style.transform = 'scale(1)', 200); }
function matchLatestVehiclesHeight() {
  const camCard = document.querySelector('.live-cam-card'); const recentCard = document.querySelector('.live-recent-card'); const wrap = document.querySelector('.vehicle-row-list-wrap'); if (!camCard || !recentCard || !wrap) return;
  const camHeight = camCard.getBoundingClientRect().height; if (camHeight < 50) return; recentCard.style.height = camHeight + 'px'; const headerHeight = recentCard.querySelector('h2')?.getBoundingClientRect().height || 0; wrap.style.maxHeight = (camHeight - headerHeight - 44) + 'px';
}
const streamImgEl = document.getElementById('stream-img');
if (streamImgEl) { streamImgEl.addEventListener('load', matchLatestVehiclesHeight); streamImgEl.addEventListener('error', function() { setTimeout(function() { streamImgEl.src = '/stream?reconnect=' + Date.now(); }, 800); }); }
window.addEventListener('resize', matchLatestVehiclesHeight); setTimeout(matchLatestVehiclesHeight, 500); setInterval(matchLatestVehiclesHeight, 3000);
async function refreshOperations() {
  try {
    const o = await (await fetch('/api/operations')).json(); const camOnline = o.camera.status === 'ONLINE'; document.getElementById('cam-offline-overlay').classList.toggle('show', !camOnline); document.getElementById('src-info').textContent = o.camera.stream ? '\u00b7 ' + o.camera.stream + ' \u2192 704x576' : '';
    document.getElementById('ops-ai-fps').innerHTML = o.ai_engine.effective_fps !== null ? '<b>' + o.ai_engine.effective_fps + ' FPS</b> (' + o.ai_engine.fps + ' detections/s)' : '<b>measuring...</b>';
    const cpuStr = o.hardware.cpu_pct !== null ? o.hardware.cpu_pct + '%' : '--'; const tStr = o.hardware.temp_c !== null ? o.hardware.temp_c + '°C' : '--'; document.getElementById('ops-cpu-gpu-pct').innerHTML = '<b>' + cpuStr + ' / ' + tStr + '</b>';
    const rd = o.recording.duration_seconds !== null ? Math.floor(o.recording.duration_seconds / 60) + 'm ' + (o.recording.duration_seconds % 60) + 's' : ''; document.getElementById('ops-recording-mode').textContent = o.recording.status + (rd ? ' · ' + rd : '');
    document.getElementById('ops-storage-pct').textContent = o.storage.used_pct !== null ? o.storage.used_pct + '% used · ' + o.storage.free_gb + ' GB free' : '--';
    document.getElementById('cr-cam-fps').textContent = o.ai_engine.effective_fps !== null ? o.ai_engine.effective_fps : '--'; document.getElementById('cr-inf-fps').textContent = o.ai_engine.fps !== null ? o.ai_engine.fps : '--';
    document.getElementById('cr-vpm').textContent = o.traffic.vehicles_per_minute; document.getElementById('cr-traffic').textContent = o.traffic.level; document.getElementById('cr-recording').textContent = o.recording.status;
    document.getElementById('cr-camera').textContent = o.camera.status; document.getElementById('cr-health').textContent = o.system_health.percent + '%';
  } catch (e) { console.error('operations refresh failed', e); }
}
let _fails = 0, _lastTop = undefined, _lastOk = Date.now();
function markOk() { _fails = 0; _lastOk = Date.now(); const b = document.getElementById('connection-lost-banner'); if (b) b.style.display = 'none'; }
function markFail() { _fails += 1; if (_fails >= 3) { const b = document.getElementById('connection-lost-banner'); if (b) b.style.display = 'block'; } }
setInterval(() => { const b = document.getElementById('connection-lost-banner'); if (b && b.style.display === 'block') b.textContent = '⚠️ LOST CONNECTION TO DEVICE - this dashboard has not heard back from the Raspberry Pi for ' + Math.floor((Date.now() - _lastOk) / 1000) + ' second(s). It may be powered off, rebooting, or its network link is down. This page keeps trying and updates automatically once contact is restored.'; }, 1000);
async function refresh() {
  try {
    const s = await (await fetch('/api/stats')).json(); markOk();
    animateNum(document.getElementById('stat-total'), s.total); animateNum(document.getElementById('stat-up'), s.up); animateNum(document.getElementById('stat-down'), s.down); renderChart(s.by_hour);
    document.getElementById('cr-total').textContent = s.total; document.getElementById('cr-up').textContent = s.up; document.getElementById('cr-down').textContent = s.down;
    const tt = Math.max(1, s.up + s.down); document.getElementById('stat-up-pct').textContent = (100 * s.up / tt).toFixed(1) + '% of traffic'; document.getElementById('stat-down-pct').textContent = (100 * s.down / tt).toFixed(1) + '% of traffic';
    const y = document.getElementById('stat-vs-yest'); if (s.vs_yesterday_pct !== null) { y.className = 'kpi-context ' + (s.vs_yesterday_pct >= 0 ? 'trend-up' : 'trend-down'); y.innerHTML = (s.vs_yesterday_pct >= 0 ? '&#8593; ' : '&#8595; ') + Math.abs(s.vs_yesterday_pct) + '% vs yesterday'; } else { y.className = 'kpi-context'; y.textContent = 'No data for yesterday yet'; }
    document.getElementById('stat-peak-hour').innerHTML = s.peak_hour !== null ? (String(s.peak_hour).padStart(2, '0') + ':00&ndash;' + String((s.peak_hour + 1) % 24).padStart(2, '0') + ':00') : '&mdash;'; document.getElementById('stat-peak-hour-count').textContent = s.peak_hour !== null ? s.peak_hour_count + ' vehicles' : 'No data yet';
    const banner = document.getElementById('camera-banner'); if (!s.camera_connected) { banner.style.display = 'block'; banner.textContent = '⚠️ Camera disconnected - reconnecting... counts are not updating right now.'; } else banner.style.display = 'none';
    const list = document.getElementById('recent-list'); const newest = s.recent.length ? s.recent[0].id : null;
    if (newest !== _lastTop) {
      _lastTop = newest;
      list.innerHTML = s.recent.length ? s.recent.map(r => {
        const conf = r.confidence !== null ? Math.round(Number(r.confidence)) + '%' : '';
        const attrs = `data-id="${r.id}" data-date="${r.date}" data-time="${r.time}" data-type="${r.type}" data-direction="${r.direction}" data-confidence="${r.confidence !== null ? r.confidence : ''}" data-snapshot="${r.snapshot || ''}"`;
        const img = r.snapshot ? `<img src="/snapshot_thumb/${r.snapshot}" class="lv-img clickable-thumb" loading="lazy" ${attrs} onerror="handleThumbError(this)">` : '<div class="lv-noimg">no image</div>';
        const up = r.direction === 'IN';
        return `<div class="lv-tile"><div class="lv-top"><span class="lv-id">#${r.id}</span><span class="lv-type" title="${r.type}">${r.type}</span></div>${img}` +
               `<div class="lv-time">${r.time}${conf ? ' <span>&mdash; ' + conf + '</span>' : ''}</div><div class="lv-badges"><span class="lv-b ok">&#10003; COUNTED</span><span class="lv-b ${up ? 'ok' : 'dn'}">${up ? '&#8593; IN' : '&#8595; OUT'}</span></div></div>`;
      }).join('') : '<div class="lv-empty">No vehicles logged yet</div>';
    }
  } catch (e) { console.error('refresh failed', e); markFail(); }
}
async function refreshTrafficIntelligence() {
  try {
    const t = await (await fetch('/api/traffic_intelligence')).json();
    document.getElementById('ti-avg-vpm').textContent = t.avg_vehicles_per_minute !== null ? t.avg_vehicles_per_minute : 'No data yet';
    document.getElementById('ti-busiest').textContent = t.busiest_15min_start !== null ? t.busiest_15min_start + ' (' + t.busiest_15min_count + ' vehicles)' : 'No data yet';
    const cmp = document.getElementById('ti-hour-cmp'); if (t.hour_vs_previous_pct !== null) { const up = t.hour_vs_previous_pct >= 0; cmp.innerHTML = `<span class="${up ? 'trend-up' : 'trend-down'}">${up ? '&#8593;' : '&#8595;'} ${Math.abs(t.hour_vs_previous_pct)}%</span>`; cmp.title = t.current_hour_count + ' this hour vs ' + t.previous_hour_count + ' last hour'; } else cmp.textContent = t.current_hour_count + ' this hour (no prior hour to compare)';
    const g = document.getElementById('ti-avg-gap'); g.textContent = (t.avg_gap_seconds !== null) ? (t.avg_gap_seconds >= 60 ? (t.avg_gap_seconds / 60).toFixed(1) + ' min' : t.avg_gap_seconds.toFixed(1) + ' sec') : 'Not enough data yet';
    const bd = document.getElementById('ti-class-breakdown'); const LIGHT = new Set(['two_wheeler', 'cycle', 'auto_rickshaw']);
    const col = (ty) => { ty = ty.toLowerCase(); if (LIGHT.has(ty)) return 'linear-gradient(90deg, #7c9eff, #a5c0ff)'; if (ty === 'car') return 'linear-gradient(90deg, #a78bfa, #c4b1fc)'; return 'linear-gradient(90deg, #fb923c, #fdba74)'; };
    if (t.class_breakdown.length) { bd.className = ''; bd.innerHTML = t.class_breakdown.map(c => `<div class="ti-class-row"><span class="ti-class-name">${c.type}</span><div class="ti-class-track"><div class="ti-class-fill" style="width:${c.pct}%; background:${col(c.type)};"></div></div><span class="ti-class-pct">${c.pct}%</span></div>`).join(''); }
    else { bd.className = 'chart-empty'; bd.textContent = 'No vehicles yet today'; }
  } catch (e) { console.error('traffic intelligence refresh failed', e); }
}
refresh(); refreshOperations(); refreshTrafficIntelligence(); setInterval(refresh, 1000); setInterval(refreshOperations, 3000); setInterval(refreshTrafficIntelligence, 10000);
</script>"""

REC_BODY = r"""
<div class="card"><div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:6px;"><h2 style="margin:0;">Current Recording</h2><button class="btn btn-primary" id="rec-toggle">Start recording</button></div>
<div class="ops-row"><span class="ops-row-label">Status</span><span class="ops-row-value"><b id="rec-status">--</b></span></div>
<div class="ops-row"><span class="ops-row-label">Filename</span><span class="ops-row-value" id="rec-filename">&mdash;</span></div>
<div class="ops-row"><span class="ops-row-label">Duration</span><span class="ops-row-value" id="rec-duration">--</span></div>
<div class="ops-row"><span class="ops-row-label">File Size</span><span class="ops-row-value" id="rec-filesize">--</span></div>
<p style="color:var(--muted); font-size:12px; margin-top:10px; margin-bottom:0;">Press Start recording to save the live annotated video (H.264, 25 FPS) to the SSD. The view selector below only filters which saved recordings are displayed; it never affects the active recording.</p></div>
<div class="card" style="margin-bottom:16px;"><h2>Recording Health Monitor <span class="sub">(real, lightweight file/timing checks)</span></h2>
<div class="ops-row"><span class="ops-row-label">Recording Health</span><span class="ops-row-value"><span class="ops-dot" id="rh-status-dot"></span> <b id="rh-status">--</b></span></div>
<div class="ops-row"><span class="ops-row-label">Current File</span><span class="ops-row-value" id="rh-current-file">--</span></div>
<div class="ops-row"><span class="ops-row-label">Current File Size</span><span class="ops-row-value" id="rh-current-size">--</span></div>
<div class="ops-row"><span class="ops-row-label">Recording FPS</span><span class="ops-row-value" id="rh-fps">--</span></div>
<div class="ops-row"><span class="ops-row-label">Last Frame Written</span><span class="ops-row-value" id="rh-last-frame">--</span></div>
<div class="ops-row"><span class="ops-row-label">Last Finalized File</span><span class="ops-row-value" id="rh-last-finalized-file">--</span></div>
<div class="ops-row"><span class="ops-row-label">Last Finalized Time</span><span class="ops-row-value" id="rh-last-finalized-time">--</span></div>
<div class="ops-row"><span class="ops-row-label">Detected Gaps</span><span class="ops-row-value" id="rh-gaps">--</span></div>
<div class="ops-row"><span class="ops-row-label">Failed Recordings</span><span class="ops-row-value" id="rh-failures">--</span></div></div>
<div class="card"><div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:12px;"><h2 style="margin:0;">Saved Recordings <span class="sub" id="rec-total">(0 total)</span></h2></div>
<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:14px; padding-bottom:14px; border-bottom:1px solid var(--border);">
  <label style="font-size:12px; color:var(--muted); font-weight:700;">RECORDING VIEW:</label>
  <select id="rec-view-select" style="padding:8px 10px; border-radius:8px; border:1px solid var(--border); background:var(--surface-2); color:var(--text);"><option value="all">All</option><option value="hourly">Hourly</option><option value="daily">Daily</option><option value="monthly">Monthly</option><option value="custom">Custom</option></select>
  <span id="rv-daily-fields" style="display:none;"><input type="date" id="rv-date"></span>
  <span id="rv-hourly-fields" style="display:none; gap:10px; align-items:center;"><input type="date" id="rv-hourly-date"><select id="rv-hour">__HOURS__</select></span>
  <span id="rv-monthly-fields" style="display:none;"><input type="month" id="rv-month"></span>
  <span id="rv-custom-fields" style="display:none; gap:8px; align-items:center; flex-wrap:wrap;"><label style="font-size:12px; color:var(--muted);">From:</label><input type="date" id="rv-from-date"><input type="time" id="rv-from-time" value="00:00"><label style="font-size:12px; color:var(--muted);">To:</label><input type="date" id="rv-to-date"><input type="time" id="rv-to-time" value="23:59"></span>
  <button class="window-btn" id="rv-apply">Apply View</button></div>
<table id="recordings-table"><tr><th>Filename</th><th>Date</th><th>Time</th><th>Size</th><th>Status</th><th>Actions</th></tr></table></div>
<div class="snap-modal-backdrop" id="vid-modal-backdrop"><div class="snap-modal" style="max-width:1000px;"><button class="snap-modal-close" id="vid-modal-close" title="Close" aria-label="Close">&#10005;</button><div class="snap-modal-img-wrap" style="flex:1 1 100%;"><video id="vid-player" controls style="max-width:100%; max-height:80vh; background:#000;"></video></div></div></div>
"""
REC_JS = r"""<script>
let recOn = false;
async function refreshRecordingStatus() {
  try {
    const o = await (await fetch('/api/operations')).json(); recOn = o.recording.status === 'ACTIVE'; document.getElementById('rec-status').textContent = o.recording.status;
    document.getElementById('rec-duration').textContent = o.recording.duration_seconds !== null ? Math.floor(o.recording.duration_seconds / 60) + 'm ' + (o.recording.duration_seconds % 60) + 's' : '--'; document.getElementById('rec-filename').textContent = o.recording.filename || '—';
    const b = document.getElementById('rec-toggle'); b.textContent = recOn ? 'Stop recording' : 'Start recording'; b.className = 'btn ' + (recOn ? 'btn-danger' : 'btn-primary');
  } catch (e) { console.error('recording status refresh failed', e); }
}
document.getElementById('rec-toggle').addEventListener('click', async () => { await fetch('/api/record', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: recOn ? 'stop' : 'start'})}); await refreshRecordingStatus(); await refreshRecordingHealth(); reloadRecordingsList(); });
function dotClass(s) { return s === 'HEALTHY' ? 'ops-dot-good' : (s === 'WARNING' ? 'ops-dot-warn' : 'ops-dot-bad'); }
async function refreshRecordingHealth() {
  try {
    const h = await (await fetch('/api/recording_health')).json(); document.getElementById('rh-status').textContent = h.status; document.getElementById('rh-status-dot').className = 'ops-dot ' + dotClass(h.status);
    document.getElementById('rh-current-file').textContent = h.current_file || '— (not currently recording)'; document.getElementById('rh-current-size').textContent = h.current_file_size_bytes !== null ? (h.current_file_size_bytes / 1024).toFixed(1) + ' KB' : 'N/A';
    document.getElementById('rec-filesize').textContent = h.current_file_size_bytes !== null ? (h.current_file_size_bytes / 1048576).toFixed(1) + ' MB' : '--';
    document.getElementById('rh-fps').textContent = h.recording_fps !== null ? h.recording_fps : 'Not Measured'; document.getElementById('rh-last-frame').textContent = h.last_frame_written_at ? new Date(h.last_frame_written_at).toLocaleTimeString() : 'N/A';
    document.getElementById('rh-last-finalized-file').textContent = h.last_finalized_file || '—'; document.getElementById('rh-last-finalized-time').textContent = h.last_finalized_time ? new Date(h.last_finalized_time).toLocaleString() : '—';
    document.getElementById('rh-gaps').textContent = h.detected_gaps.length + ' detected'; document.getElementById('rh-failures').textContent = h.failed_recordings.length + ' failed';
  } catch (e) { console.error('recording health refresh failed', e); }
}
function renderRecordingRow(r) {
  let actions;
  if (r.status === 'RECORDING') actions = '<span style="color:var(--muted); font-size:12px;">Recording in progress</span>';
  else actions = `<a href="#" class="dl-btn rec-play-btn" data-filename="${r.filename}">Play</a> <a href="/video/${r.filename}?download=1" class="dl-btn">Download</a> `;
  actions += `<button class="window-btn rec-delete-btn" data-filename="${r.filename}"${r.status === 'RECORDING' ? ' disabled' : ''}>Delete</button>`;
  return `<tr><td>${r.filename}</td><td>${r.date}</td><td>${r.time}</td><td>${r.size_mb} MB</td><td><span class="pill ${r.status === 'READY' ? 'up-pill' : 'down-pill'}">${r.status}</span></td><td>${actions}</td></tr>`;
}
function updateViewFieldVisibility() {
  const view = document.getElementById('rec-view-select').value; document.getElementById('rv-daily-fields').style.display = view === 'daily' ? 'inline' : 'none'; document.getElementById('rv-hourly-fields').style.display = view === 'hourly' ? 'inline-flex' : 'none';
  document.getElementById('rv-monthly-fields').style.display = view === 'monthly' ? 'inline' : 'none'; document.getElementById('rv-custom-fields').style.display = view === 'custom' ? 'inline-flex' : 'none';
}
document.getElementById('rec-view-select').addEventListener('change', updateViewFieldVisibility); updateViewFieldVisibility();
async function reloadRecordingsList() {
  const view = document.getElementById('rec-view-select').value; let url = '/api/recordings';
  if (view === 'daily') { const d = document.getElementById('rv-date').value; if (d) url += '?date=' + d; }
  else if (view === 'hourly') { const d = document.getElementById('rv-hourly-date').value, h = document.getElementById('rv-hour').value; if (d) url += '?view=hourly&date=' + d + '&hour=' + h; }
  else if (view === 'monthly') { const m = document.getElementById('rv-month').value; if (m) url += '?month=' + m; }
  else if (view === 'custom') { const fd = document.getElementById('rv-from-date').value, ft = document.getElementById('rv-from-time').value || '00:00', td = document.getElementById('rv-to-date').value, tt = document.getElementById('rv-to-time').value || '23:59'; if (fd && td) url += '?view=custom&from_date=' + fd + '&from_time=' + ft + '&to_date=' + td + '&to_time=' + tt; }
  const data = await (await fetch(url)).json(); document.getElementById('rec-total').textContent = '(' + data.total + ' total)';
  document.getElementById('recordings-table').innerHTML = '<tr><th>Filename</th><th>Date</th><th>Time</th><th>Size</th><th>Status</th><th>Actions</th></tr>' + (data.recordings.length ? data.recordings.map(renderRecordingRow).join('') : '<tr><td colspan=6 style="color:var(--muted); text-align:center; padding:30px;">No recordings match this view</td></tr>');
}
document.getElementById('rv-apply').addEventListener('click', reloadRecordingsList);
function closeVid() { document.getElementById('vid-modal-backdrop').classList.remove('show'); const p = document.getElementById('vid-player'); p.pause(); p.removeAttribute('src'); p.load(); }
document.getElementById('vid-modal-close').addEventListener('click', closeVid); document.getElementById('vid-modal-backdrop').addEventListener('click', (e) => { if (e.target.id === 'vid-modal-backdrop') closeVid(); }); document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeVid(); });
document.addEventListener('click', async (e) => {
  const pl = e.target.closest('.rec-play-btn'); if (pl) { e.preventDefault(); const p = document.getElementById('vid-player'); p.src = '/video/' + pl.dataset.filename; document.getElementById('vid-modal-backdrop').classList.add('show'); p.play().catch(() => {}); return; }
  if (!e.target.classList.contains('rec-delete-btn')) return; const filename = e.target.dataset.filename; if (!confirm('Permanently delete ' + filename + '? This cannot be undone.')) return;
  const data = await (await fetch('/api/recordings/delete', {method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: 'filename=' + encodeURIComponent(filename)})).json();
  if (data.success) reloadRecordingsList(); else alert('Delete failed: ' + (data.error || 'unknown error'));
});
refreshRecordingStatus(); refreshRecordingHealth(); reloadRecordingsList(); setInterval(refreshRecordingStatus, 3000); setInterval(refreshRecordingHealth, 3000); setInterval(reloadRecordingsList, 8000);
</script>"""

WIDGET_CSS = """<style>
.vclass-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; } .vclass-chip { display:flex; align-items:center; gap:6px; font-size:11px; font-weight:600; color:var(--ink-dim); background:var(--surface-2); padding:5px 10px 5px 6px; border-radius:20px; border:1px solid var(--border); cursor:pointer; }
.vclass-swatch { width:9px; height:9px; border-radius:3px; flex-shrink:0; } .vr-filter-row { display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:4px; } .vr-filter-row > div { display:flex; flex-direction:column; gap:4px; }
.vr-filter-row label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; font-weight:700; } .vr-filter-row input, .vr-filter-row select { padding:7px 9px; border-radius:9px; border:1px solid var(--border); background:var(--surface-2); color:var(--text); font-size:12px; }
.vr-btn-apply { background:linear-gradient(135deg, #7c9eff, #5b7fe8); color:#fff; border:none; padding:9px 18px; border-radius:9px; font-weight:700; font-size:12.5px; cursor:pointer; box-shadow:0 4px 12px -4px rgba(91,127,232,0.5); }
.vr-btn-clear { background:var(--surface); color:var(--text); border:1px solid var(--border); padding:9px 18px; border-radius:9px; font-weight:700; font-size:12.5px; text-decoration:none; display:inline-block; }
.vr-table-scroll { overflow-x:auto; } .vr-table-scroll table { width:100%; } .vr-table-scroll th, .vr-table-scroll td { white-space:nowrap; font-size:12.5px; } .vr-sort-link { color:inherit; text-decoration:none; } .vr-sort-link:hover { color:var(--lime); }
.vr-thumb { width:165px; height:95px; border-radius:8px; object-fit:contain; background:var(--surface-2); border:1px solid var(--border); } .vr-thumb-empty { display:flex; align-items:center; justify-content:center; background:var(--surface-2); color:var(--muted); font-size:16px; }
.page-btn { display:inline-flex; align-items:center; justify-content:center; padding:6px 12px; border-radius:8px; border:1px solid var(--border); background:var(--surface); color:var(--text); font-size:12px; font-weight:700; text-decoration:none; margin-right:4px; }
.page-btn.active { background:var(--lime); color:var(--lime-ink); border-color:transparent; } .page-btn.disabled { opacity:0.4; pointer-events:none; } .vr-per-page { padding:6px 8px; border-radius:8px; border:1px solid var(--border); background:var(--surface-2); color:var(--text); font-size:12px; }
</style>"""


def widget(q, action):
    g = lambda k, d="": (q.get(k) or [d])[0]; vtypes = q.get("vtype", []); direction = g("direction", "all"); sort = g("sort", "timestamp"); sd = "asc" if g("sort_dir") == "asc" else "desc"
    try: per_page = int(g("per_page", "25")); per_page = per_page if per_page in (10, 25, 50, 100) else 25
    except ValueError: per_page = 25
    try: page = max(1, int(g("page", "1")))
    except ValueError: page = 1
    total, rows = vehicle_query(q, per_page, (page - 1) * per_page, sort, sd); pages = max(1, (total + per_page - 1) // per_page); first = 0 if not total else (page - 1) * per_page + 1; last = min(page * per_page, total)

    def row(r):
        rid, day, ts, cls, d, conf, snap = r; t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        thumb = ('<img src="/snapshot_thumb/%d.jpg" class="vr-thumb" loading="lazy" alt="snapshot">' % rid) if snap else '<div class="vr-thumb vr-thumb-empty">&#128247;</div>'
        return '<tr><td>%d</td><td>%s<br><span style="color:var(--muted)">%s</span></td><td>%s</td><td>%s</td><td><span class="pill %s">%s %s</span></td><td>%s</td></tr>' % (rid, day, t, esc(cls), thumb, "up-pill" if d == "IN" else "down-pill", "&#8593;" if d == "IN" else "&#8595;", d, ("%.2f%%" % (conf * 100)) if conf is not None else "--")
    body = "".join(row(r) for r in rows) or '<tr><td colspan="6" style="color:var(--muted); text-align:center; padding:24px;">No vehicles found for these filters</td></tr>'
    sw = lambda c: "#7c9eff" if c.lower() in LIGHT else ("#a78bfa" if c.lower() == "car" else "#fb923c")
    chips = "".join('<label class="vclass-chip"><input type="checkbox" name="vtype" value="%s" %s><span class="vclass-swatch" style="background:%s"></span>%s</label>' % (esc(c), "checked" if c in vtypes else "", sw(c), esc(c)) for c in S["cfg"].get("classes", []))

    def qs(extra):
        d = {k: v[0] for k, v in q.items() if k != "vtype" and v}; d.update(extra); return urlencode(d) + "".join("&vtype=" + quote(v) for v in vtypes)
    def sortlink(col, label): nd = "asc" if (sort == col and sd == "desc") else "desc"; return '<a href="%s?%s" class="vr-sort-link">%s%s</a>' % (action, qs({"sort": col, "sort_dir": nd}), label, "" if sort != col else (" &#9650;" if sd == "asc" else " &#9660;"))
    def plink(p, label=None): return '<a class="page-btn%s%s" href="%s?%s">%s</a>' % (" active" if p == page else "", " disabled" if p < 1 or p > pages else "", action, qs({"page": str(p)}), label or p)
    links = ([plink(max(1, page - 1), "Previous")] + [plink(p) for p in range(max(1, page - 2), min(pages, page + 2) + 1)] + [plink(min(pages, page + 1), "Next")]) if pages > 1 else []
    sel = lambda v: "selected" if direction == v else ""
    return WIDGET_CSS + ('<div class="card" style="margin-bottom:14px;"><form method="get" action="%s"><div class="vclass-row">%s</div><div class="vr-filter-row"><div><label>Search (Vehicle ID)</label><input type="text" name="search" value="%s" placeholder="e.g. 57"></div>'
            '<div><label>Direction</label><select name="direction"><option value="all" %s>All</option><option value="IN" %s>In</option><option value="OUT" %s>Out</option></select></div>'
            '<div><label>Start Date &amp; Time</label><div style="display:flex; gap:4px;"><input type="date" name="start_date" value="%s"><input type="time" name="start_time" value="%s"></div></div>'
            '<div><label>End Date &amp; Time</label><div style="display:flex; gap:4px;"><input type="date" name="end_date" value="%s"><input type="time" name="end_time" value="%s"></div></div>'
            '<div><button type="submit" class="vr-btn-apply">Apply</button></div><div><a class="vr-btn-clear" href="%s">Clear</a></div></div></form></div>'
            '<div class="card"><div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:10px;"><div style="display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--ink-dim);">Show '
            '<form method="get" action="%s" style="display:inline;"><select class="vr-per-page" name="per_page" onchange="this.form.submit()">%s</select></form> entries</div></div><div class="vr-table-scroll"><table><tr><th>%s</th><th>%s</th><th>%s</th><th>Object</th><th>%s</th><th>%s</th></tr>%s</table></div>'
            '<div style="margin-top:16px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;"><div style="font-size:12px; color:var(--muted);">Showing %d to %d of %d entries &middot; <span id="vr-autorefresh-note">auto-refreshing every 15s</span></div><div>%s</div></div></div>'
            '<script>let vrSecondsLeft = 15; setInterval(() => { vrSecondsLeft -= 1; const n = document.getElementById("vr-autorefresh-note"); if (n) n.textContent = "auto-refreshing in " + vrSecondsLeft + "s"; if (vrSecondsLeft <= 0) location.reload(); }, 1000);</script>') % (
        action, chips, esc(g("search")), sel("all"), sel("IN"), sel("OUT"), esc(g("start_date")), esc(g("start_time")), esc(g("end_date")), esc(g("end_time")), action, action,
        "".join('<option value="%d" %s>%d</option>' % (n, "selected" if per_page == n else "", n) for n in (10, 25, 50, 100)), sortlink("id", "ID"), sortlink("timestamp", "Date Time"), sortlink("vehicle_type", "AI Class"), sortlink("direction", "Direction"), sortlink("confidence", "Confidence"), body, first, last, total, "".join(links))


def page_custom(q):
    g = lambda k, d="": (q.get(k) or [d])[0]; start, end, st, et, vt, dr = g("start"), g("end"), g("start_time"), g("end_time"), g("type", "all"), g("direction", "all"); result = ""
    if start and not DATE.match(start): result = '<div class="card"><h2>Result</h2><p style="color:var(--bad);">Invalid start date format.</p></div>'; start = ""
    elif end and not DATE.match(end): result = '<div class="card"><h2>Result</h2><p style="color:var(--bad);">Invalid end date format.</p></div>'; start = ""
    if start:
        e2 = end or start; _, n = report_xlsx(_ts(start, st if TIME.match(st) else "00:00"), _ts(e2, et if TIME.match(et) else "23:59") + 59, vt, dr)
        result = '<div class="card"><h2>Result</h2><p style="color:var(--muted)">%d vehicle(s) found for %s%s.</p><a class="dl-btn" href="/download/custom?%s">Download Excel Report</a></div>' % (n, esc(start), (" to " + esc(e2)) if e2 != start else "", urlencode({"start": start, "end": e2, "start_time": st, "end_time": et, "type": vt, "direction": dr}))
    types = "".join('<option value="%s" %s>%s</option>' % (esc(c), "selected" if c == vt else "", esc(c)) for c in S["cfg"].get("classes", [])); lbl = '<label style="display:block; font-size:11px; color:var(--muted); margin-bottom:4px;">%s</label>'
    s2 = 'style="padding:9px 10px; border-radius:8px; border:1px solid var(--border); background:var(--surface-2); color:var(--text);"'; dsel = lambda v: "selected" if dr == v else ""
    body = ('<div class="card" style="margin-bottom:16px;"><h2>Quick Links</h2><div style="display:flex; gap:10px; flex-wrap:wrap;"><a class="dl-btn" href="/hourly">Hourly Reports</a><a class="dl-btn" href="/daily">Daily Reports</a><a class="dl-btn" href="/monthly">Monthly Reports</a></div></div>'
            '<div class="card" style="margin-bottom:16px;"><h2>Custom Report Builder</h2><form method="get" action="/custom"><div style="display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:10px;">'
            '<div>%s<input type="date" name="start" value="%s" required></div><div>%s<input type="date" name="end" value="%s"></div><div>%s<input type="time" name="start_time" value="%s"></div><div>%s<input type="time" name="end_time" value="%s"></div>'
            '<div>%s<select name="type" %s><option value="all">All Types</option>%s</select></div><div>%s<select name="direction" %s><option value="all" %s>All</option><option value="IN" %s>In</option><option value="OUT" %s>Out</option></select></div></div>'
            '<input type="submit" value="Generate Report"></form></div>%s<h2 style="margin:20px 0 12px; font-size:13px; color:var(--text); font-weight:800; text-transform:uppercase; letter-spacing:0.6px;">Live Vehicle Report</h2>%s') % (
        lbl % "Start Date", esc(start), lbl % "End Date (optional)", esc(end), lbl % "From Time", esc(st), lbl % "To Time", esc(et), lbl % "Vehicle Type", s2, types, lbl % "Direction", s2, dsel("all"), dsel("IN"), dsel("OUT"), result, widget(q, "/custom"))
    return shell("reports", "Reports", "Custom Report Builder", body)


def page_report_list(title, kind, prefix):
    rows = "".join('<tr><td>%s</td><td><a class="dl-btn" href="/view-report?start_date=%s&start_time=%s&end_date=%s&end_time=%s" style="background:var(--surface-2); color:var(--text);">&#128065; View</a></td><td><a class="dl-btn" href="%s/%s">Download</a></td></tr>' % ((esc(f),) + report_range(f)[2] + (prefix, f)) for f in report_files(kind)) or '<tr><td colspan=3 style="color:var(--muted)">No reports generated yet for this period</td></tr>'
    return shell("reports", title, "", '<div class="card"><h2>%s</h2><table><tr><th>File</th><th>View</th><th>Download</th></tr>%s</table></div>' % (esc(title), rows))


# ---------------------------------------------------------------------------------------------------- HTTP
CTYPES = {".css": "text/css; charset=utf-8", ".png": "image/png", ".woff2": "font/woff2", ".js": "text/javascript"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str): body = body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _json(self, o): self._send(200, json.dumps(o))

    def do_GET(self):
        u = urlparse(self.path); p = u.path; q = parse_qs(u.query)
        try:
            if p in ("/", ""): self._send(200, shell("live", "Live Monitor", "Camera 01 · Main Road", LIVE_BODY, LIVE_JS), "text/html; charset=utf-8")
            elif p == "/recordings": self._send(200, shell("recordings", "Recordings", "Video recording management", REC_BODY.replace("__HOURS__", "".join('<option value="%d">%02d:00</option>' % (h, h) for h in range(24))), REC_JS), "text/html; charset=utf-8")
            elif p == "/custom": self._send(200, page_custom(q), "text/html; charset=utf-8")
            elif p in ("/hourly", "/daily", "/monthly"): self._send(200, page_report_list({"/hourly": "Hourly Reports", "/daily": "Daily Reports", "/monthly": "Monthly Reports"}[p], p[1:], "/download/" + p[1:]), "text/html; charset=utf-8")
            elif p == "/view-report": self._send(200, shell("reports", "View Report", "Real-time vehicle log viewer", widget(q, "/view-report")), "text/html; charset=utf-8")
            elif p == "/favicon.ico": self._send(204, b"", "image/x-icon")
            elif p.startswith("/static/"): self._static(p[len("/static/"):])
            elif p == "/api/stats": self._json(api_stats())
            elif p == "/api/operations": self._json(api_operations())
            elif p == "/api/traffic_intelligence": self._json(api_intel())
            elif p == "/api/recordings": self._json(api_recordings(q))
            elif p == "/api/recording_health": self._json(api_rec_health())
            elif p in ("/stream", "/stream.mjpg"): self._stream()
            elif p.startswith(("/snapshot/", "/snapshot_thumb/", "/snap/")) and p.endswith(".jpg"): self._snap(p.rsplit("/", 1)[1][:-4])
            elif p.startswith("/video/"): self._file(os.path.join(S["cfg"]["rec_dir"], p[len("/video/"):]), p[len("/video/"):], "download" in q)
            elif p.startswith("/recordings/"): self._file(os.path.join(S["cfg"]["rec_dir"], p[len("/recordings/"):]), p[len("/recordings/"):], False)
            elif p.startswith("/download/") and p.count("/") == 3 and p.split("/")[2] in ("hourly", "daily", "monthly"):
                name = p.rsplit("/", 1)[1]; rg = report_range(name) if SAFE.match(name) else None
                if rg and name.startswith(p.split("/")[2]): data, _ = report_xlsx(rg[0], rg[1], title=name); self._xlsx(name, data)
                else: self._send(404, '{"error":"not found"}')
            elif p == "/download/custom":
                g = lambda k, d="": (q.get(k) or [d])[0]; s = g("start")
                if not DATE.match(s): return self._send(400, '{"error":"bad date"}')
                e = g("end") if DATE.match(g("end")) else s; data, _ = report_xlsx(_ts(s, g("start_time") if TIME.match(g("start_time")) else "00:00"), _ts(e, g("end_time") if TIME.match(g("end_time")) else "23:59") + 59, g("type", "all"), g("direction", "all")); self._xlsx("custom_%s_to_%s.xlsx" % (s, e), data)
            else: self._send(404, '{"error":"not found"}')
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as e:
            try: self._send(500, json.dumps({"error": str(e)}))
            except Exception: pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); raw = self.rfile.read(n) if n else b""; body = {}
        try: body = json.loads(raw or b"{}") if "json" in (self.headers.get("Content-Type") or "") else {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "ignore")).items()}
        except ValueError: body = {}
        try:
            if self.path == "/api/record": act = body.get("action"); ok = start_rec() if act == "start" else ((stop_rec() or True) if act == "stop" else False); self._json({"ok": bool(ok)})
            elif self.path == "/api/reset": S["cmd_q"].put(("reset",)); self._json({"ok": True})
            elif self.path == "/api/recordings/delete":
                name = str(body.get("filename") or body.get("name") or ""); p = os.path.join(S["cfg"]["rec_dir"], name); busy = S["rec"]["on"] and S["rec"]["file"] == name
                if SAFE.match(name) and name.lower().endswith((".mp4", ".webm")) and os.path.isfile(p) and not busy: os.remove(p); self._json({"success": True, "ok": True})
                else: self._send(400, '{"success":false,"ok":false,"error":"cannot delete this file"}')
            else: self._send(404, '{"error":"not found"}')
        except (BrokenPipeError, ConnectionResetError): pass

    def _xlsx(self, name, data): self._send(200, data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", {"Content-Disposition": 'attachment; filename="%s"' % name})

    def _static(self, rel):
        p = os.path.realpath(os.path.join(ASSETS, rel))
        if not p.startswith(os.path.realpath(ASSETS) + os.sep) or not os.path.isfile(p): return self._send(404, '{"error":"not found"}')
        with open(p, "rb") as fh: self._send(200, fh.read(), CTYPES.get(os.path.splitext(p)[1], "application/octet-stream"), {"Cache-Control": "max-age=86400"})

    def _snap(self, sid):
        if not sid.isdigit(): return self._send(404, '{"error":"not found"}')
        rows, _ = dbq("SELECT snap FROM crossings WHERE id=?", (int(sid),)); p = rows[0][0] if rows else None
        if p and os.path.isfile(p) and os.path.realpath(p).startswith(os.path.realpath(S["cfg"]["data_dir"])):
            with open(p, "rb") as fh: self._send(200, fh.read(), "image/jpeg", {"Cache-Control": "max-age=86400"})
        else: self._send(404, '{"error":"not found"}')

    def _stream(self):
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-store"); self.end_headers(); last = 0; S["clients"] += 1
        try:
            while not S["stop"].is_set():
                with S["cond"]:
                    if S["jpg_id"] == last: S["cond"].wait(timeout=1.0)
                    jpg, cur = S["jpg"], S["jpg_id"]
                if jpg is None or cur == last: continue
                last = cur; self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError): pass
        finally: S["clients"] -= 1

    def _file(self, path, name, download):
        if not SAFE.match(name) or not os.path.isfile(path): return self._send(404, '{"error":"not found"}')
        size = os.path.getsize(path); a, b, code = 0, size - 1, 200; rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            try: a = int(s) if s else 0; b = int(e) if e else size - 1; code = 206
            except ValueError: a, b, code = 0, size - 1, 200
        b = min(b, size - 1); self.send_response(code); self.send_header("Content-Type", "video/mp4" if name.endswith(".mp4") else "video/webm"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(b - a + 1))
        if code == 206: self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
        if download: self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(a); left = b - a + 1
            while left > 0:
                chunk = fh.read(min(65536, left))
                if not chunk: break
                self.wfile.write(chunk); left -= len(chunk)


def web_main(shm, shape, seq, stop, status_q, cmd_q, cfg):
    """Entry point of the web process."""
    S["cfg"] = cfg; S["cmd_q"] = cmd_q; S["stop"] = stop; os.makedirs(cfg["rec_dir"], exist_ok=True); db_init(); buf = np.frombuffer(shm, np.uint8).reshape(shape); N = shape[0]; last = 0; ppid = os.getppid()
    try: srv = ThreadingHTTPServer((cfg["host"], cfg["port"]), Handler)
    except OSError as e: print(f"[web] cannot start the web server on port {cfg['port']}: {e}", flush=True); return
    cpu_percent(); srv.daemon_threads = True; threading.Thread(target=srv.serve_forever, daemon=True).start(); print(f"[web] pages are up on port {cfg['port']}", flush=True)
    period = 1.0 / max(cfg.get("stream_fps", 20), 1); t_enc = 0.0; t_rec = 0.0; enc_id = 0
    try:
        while not stop.is_set():
            if os.getppid() != ppid: break                                      # the detector is gone: do not stay behind holding the port
            for _ in range(100):                                                # take what the detector sent
                try: kind, payload = status_q.get_nowait()
                except Exception: break
                if kind == "status": S["status"] = payload; S["status_t"] = time.time()
                elif kind == "event":
                    try: add_event(payload)
                    except Exception as e: print(f"[web] could not store a crossing: {e}", flush=True)
            cur = seq.value; now = time.perf_counter()
            if cur != last: last = cur; S["latest"] = buf[cur % N].copy(); S["latest_id"] = cur; S["latest_t"] = time.time()
            r = S["rec"]
            if r["on"] and S["latest"] is not None and now >= t_rec:            # a fixed 25 pictures a second, so the saved video plays at the right speed
                t_rec = max(t_rec + 0.04, now - 0.2) if t_rec else now + 0.04
                if time.time() - S["latest_t"] > 1.5: r["gaps"] += 1 if (time.time() - r["last_write"]) > 1.5 else 0
                with S["lock"]:
                    if r["writer"] is not None: r["writer"].write(S["latest"]); r["frames"] += 1; r["last_write"] = time.time()
            elif not r["on"]: t_rec = 0.0
            if S["clients"] > 0 and S["latest"] is not None and S["latest_id"] != enc_id and now >= t_enc:
                ok, jpg = cv2.imencode(".jpg", S["latest"], [cv2.IMWRITE_JPEG_QUALITY, cfg.get("jpeg_quality", 75)])
                if ok:
                    enc_id = S["latest_id"]; t_enc = now + period
                    with S["cond"]: S["jpg"] = jpg.tobytes(); S["jpg_id"] += 1; S["cond"].notify_all()
            time.sleep(0.004)
    finally:
        stop_rec()
        try: srv.shutdown()
        except Exception: pass
