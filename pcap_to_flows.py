"""
PCAP → Flow aggregator.

Converts a raw PCAP file into one-row-per-flow statistics that match
the UNSW-NB15 feature schema used by heuristic_threat_model.py.

Usage:
    python3 pcap_to_flows.py <file.pcap> [output.csv]

The aggregation groups every packet into a bidirectional 5-tuple flow:
    (ip_src, ip_dst, src_port, dst_port, proto)
where (src, sport) is always the side that sent the first packet.

Calculated features:
    dur       – flow duration in seconds
    proto     – IP protocol number (6=TCP, 17=UDP, 1=ICMP)
    service   – inferred from destination port
    state     – TCP state inferred from flags (FIN/RST/INT/CON/REQ)
    spkts     – packets sent by originator
    dpkts     – packets sent by responder
    sbytes    – bytes sent by originator
    dbytes    – bytes sent by responder
    rate      – total packets / duration (pps)
    sload     – originator bits per second
    dload     – responder bits per second
    sinpkt    – mean inter-packet gap (originator), ms
    dinpkt    – mean inter-packet gap (responder), ms
    smean     – mean originator packet size (bytes)
    dmean     – mean responder packet size (bytes)
    tcprtt    – TCP RTT estimate: synack + ackdat
    synack    – time from SYN to SYN-ACK
    ackdat    – time from SYN-ACK to first ACK
"""

import csv
import io
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ── Port → service name ───────────────────────────────────────────────────────
PORT_SERVICE = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet",
    25: "smtp", 53: "dns", 67: "dhcp", 68: "dhcp",
    80: "http", 110: "pop3", 143: "imap", 161: "snmp",
    179: "bgp", 389: "ldap", 443: "ssl", 445: "microsoft-ds",
    514: "syslog", 636: "ldaps", 993: "imaps", 995: "pop3s",
    1433: "mssql", 3306: "mysql", 3389: "rdp",
    6667: "irc", 8080: "http", 8443: "ssl",
}

# ── TCP flag masks ────────────────────────────────────────────────────────────
F_FIN = 0x01
F_SYN = 0x02
F_RST = 0x04
F_PSH = 0x08
F_ACK = 0x10


def _to_port(raw) -> "int | None":
    """Convert a tshark port value (float, str, or NaN) to int or None."""
    if raw is None:
        return None
    try:
        f = float(raw)
        if f != f:   # NaN check
            return None
        v = int(f)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_flags(raw) -> int:
    try:
        if isinstance(raw, str):
            return int(raw, 16) if raw.startswith("0x") else int(raw)
        return int(raw) if raw else 0
    except (ValueError, TypeError):
        return 0


def _service(dport: Optional[int]) -> str:
    if dport is None:
        return "-"
    return PORT_SERVICE.get(dport, "-")


# ── Per-flow accumulator ──────────────────────────────────────────────────────

@dataclass
class Flow:
    key: tuple          # (src_ip, dst_ip, sport, dport, proto)
    t_first: float = 0.0
    t_last:  float = 0.0

    # originator side (whoever sent the first packet)
    s_pkts:  int   = 0
    s_bytes: int   = 0
    s_times: list  = field(default_factory=list)

    # responder side
    d_pkts:  int   = 0
    d_bytes: int   = 0
    d_times: list  = field(default_factory=list)

    # TCP handshake timing
    t_syn:     Optional[float] = None
    t_synack:  Optional[float] = None
    t_ack:     Optional[float] = None

    # TCP state flags seen
    has_fin: bool = False
    has_rst: bool = False
    has_syn: bool = False
    has_ack: bool = False

    dport: Optional[int] = None
    proto: int = 0


def _add_packet(flow: Flow, ts: float, length: int, flags: int, is_src_side: bool):
    if is_src_side:
        flow.s_pkts  += 1
        flow.s_bytes += length
        flow.s_times.append(ts)
    else:
        flow.d_pkts  += 1
        flow.d_bytes += length
        flow.d_times.append(ts)

    flow.t_last = max(flow.t_last, ts)

    # update flag state
    if flags & F_FIN:
        flow.has_fin = True
    if flags & F_RST:
        flow.has_rst = True
    if flags & F_SYN:
        flow.has_syn = True
    if flags & F_ACK:
        flow.has_ack = True

    # TCP handshake timing
    if flags & F_SYN and not (flags & F_ACK) and flow.t_syn is None:
        flow.t_syn = ts
    if flags & F_SYN and flags & F_ACK and flow.t_synack is None:
        flow.t_synack = ts
    if flags & F_ACK and not (flags & F_SYN) and flow.t_synack is not None and flow.t_ack is None:
        flow.t_ack = ts


def _finalise(flow: Flow) -> dict:
    dur = max(flow.t_last - flow.t_first, 1e-6)

    smean = (flow.s_bytes / flow.s_pkts) if flow.s_pkts else 0
    dmean = (flow.d_bytes / flow.d_pkts) if flow.d_pkts else 0

    rate  = (flow.s_pkts + flow.d_pkts) / dur
    sload = (flow.s_bytes * 8) / dur
    dload = (flow.d_bytes * 8) / dur

    def _mean_ipkt(times):
        if len(times) < 2:
            return 0.0
        gaps = [(times[i+1] - times[i]) * 1000 for i in range(len(times)-1)]
        return sum(gaps) / len(gaps)

    sinpkt = _mean_ipkt(flow.s_times)
    dinpkt = _mean_ipkt(flow.d_times)

    synack = (flow.t_synack - flow.t_syn)   if (flow.t_syn and flow.t_synack) else 0.0
    ackdat = (flow.t_ack    - flow.t_synack) if (flow.t_synack and flow.t_ack) else 0.0
    tcprtt = synack + ackdat

    # state
    if flow.has_rst:
        state = "RST"
    elif flow.has_fin:
        state = "FIN"
    elif flow.proto == 6 and flow.has_syn and not flow.has_ack:
        state = "INT"
    elif flow.proto == 6 and flow.has_syn and flow.has_ack:
        state = "CON"
    elif flow.proto == 17:
        state = "CON"
    else:
        state = "INT"

    src_ip, dst_ip, sport, dport, proto = flow.key

    return {
        "src_ip":   src_ip,
        "dst_ip":   dst_ip,
        "sport":    sport or 0,
        "dport":    dport or 0,
        "proto":    proto,
        "service":  _service(flow.dport),
        "state":    state,
        "dur":      round(dur, 6),
        "spkts":    flow.s_pkts,
        "dpkts":    flow.d_pkts,
        "sbytes":   flow.s_bytes,
        "dbytes":   flow.d_bytes,
        "rate":     round(rate, 4),
        "sload":    round(sload, 4),
        "dload":    round(dload, 4),
        "sinpkt":   round(sinpkt, 4),
        "dinpkt":   round(dinpkt, 4),
        "smean":    round(smean, 2),
        "dmean":    round(dmean, 2),
        "synack":   round(synack, 6),
        "ackdat":   round(ackdat, 6),
        "tcprtt":   round(tcprtt, 6),
    }


# ── tshark detection ─────────────────────────────────────────────────────────

def _find_tshark() -> str:
    """Locate the tshark executable on Windows, macOS, or Linux."""
    # Check PATH first (works on all platforms if Wireshark is in PATH)
    found = shutil.which("tshark")
    if found:
        return found

    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Wireshark\tshark.exe",
            r"C:\Program Files (x86)\Wireshark\tshark.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise RuntimeError(
            "tshark not found.\n"
            "Install Wireshark from https://www.wireshark.org/download.html\n"
            "Then restart your terminal (or add Wireshark to PATH)."
        )

    if platform.system() == "Darwin":
        mac_paths = [
            "/usr/local/bin/tshark",
            "/opt/homebrew/bin/tshark",
            "/Applications/Wireshark.app/Contents/MacOS/tshark",
        ]
        for path in mac_paths:
            if os.path.exists(path):
                return path

    raise RuntimeError(
        "tshark not found. Install with:\n"
        "  Linux : sudo apt install tshark\n"
        "  Mac   : brew install wireshark\n"
        "  Windows: https://www.wireshark.org/download.html"
    )


TSHARK_EXE = None   # resolved lazily on first use

TSHARK_FIELDS = [
    "frame.time_epoch", "ip.src", "ip.dst", "ip.proto",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "frame.len", "tcp.flags",
]


def extract_packets(pcap_path: str) -> pd.DataFrame:
    global TSHARK_EXE
    if TSHARK_EXE is None:
        TSHARK_EXE = _find_tshark()

    cmd = [TSHARK_EXE, "-r", pcap_path, "-T", "fields",
           "-E", "header=y", "-E", "separator=,",
           "-E", "quote=d", "-E", "occurrence=f"]
    for field in TSHARK_FIELDS:
        cmd += ["-e", field]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        # Windows: prevent a console window from flashing
        creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tshark failed: {result.stderr}")
    # dtype=str prevents pandas from auto-parsing frame.time_epoch as datetime
    # (pandas 2.x infers datetime from column names containing "time")
    df = pd.read_csv(io.StringIO(result.stdout), dtype=str)
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_flows(packets: pd.DataFrame) -> pd.DataFrame:
    flows: dict[tuple, Flow] = {}
    # track first-seen direction per flow key so we can assign src/dst correctly
    first_dir: dict[tuple, tuple] = {}

    for _, row in packets.iterrows():
        raw_ts = row.get("frame.time_epoch") or "0"
        try:
            ts = float(raw_ts)
        except (ValueError, TypeError):
            continue

        src = str(row.get("ip.src") or "").strip()
        dst = str(row.get("ip.dst") or "").strip()
        if not src or not dst or src in ("", "nan") or dst in ("", "nan"):
            continue

        proto_raw = row.get("ip.proto")
        if proto_raw is None or str(proto_raw).strip() in ("", "nan"):
            continue
        try:
            proto = int(float(proto_raw))
        except (ValueError, TypeError):
            continue
        length = int(float(row.get("frame.len") or 0))
        flags  = _parse_flags(row.get("tcp.flags"))

        # port
        if proto == 6:
            sport = _to_port(row.get("tcp.srcport"))
            dport = _to_port(row.get("tcp.dstport"))
        elif proto == 17:
            sport = _to_port(row.get("udp.srcport"))
            dport = _to_port(row.get("udp.dstport"))
        else:
            sport, dport = None, None

        # normalise flow key (always put lower IP/port first so A↔B == B↔A)
        if (src, sport) <= (dst, dport or 0):
            fwd_key = (src, dst, sport, dport, proto)
        else:
            fwd_key = (dst, src, dport, sport, proto)

        if fwd_key not in flows:
            flows[fwd_key] = Flow(
                key=fwd_key,
                t_first=ts,
                t_last=ts,
                proto=proto,
                dport=dport,
            )
            first_dir[fwd_key] = (src, sport)

        flow = flows[fwd_key]
        is_src_side = (src == first_dir[fwd_key][0])

        # update dport with the lower-numbered port (likely the server)
        if dport and (flow.dport is None or dport < flow.dport):
            flow.dport = dport

        _add_packet(flow, ts, length, flags, is_src_side)

    rows = [_finalise(f) for f in flows.values()]
    return pd.DataFrame(rows)


# ── Public entry point ────────────────────────────────────────────────────────

def pcap_to_flows(pcap_path: str) -> pd.DataFrame:
    """Convert a PCAP file to a DataFrame of flow-level features."""
    packets = extract_packets(pcap_path)
    if packets.empty:
        return pd.DataFrame()
    return aggregate_flows(packets)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pcap_to_flows.py <file.pcap> [output.csv]")
        sys.exit(1)

    pcap_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        base     = os.path.splitext(pcap_path)[0]
        out_path = base + "_flows.csv"

    print(f"Reading {pcap_path} ...")
    df = pcap_to_flows(pcap_path)
    print(f"  Extracted {len(df):,} flows")
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    print(df.head(3).to_string())
