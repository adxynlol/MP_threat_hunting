"""
CSV Packet → Flow aggregator for Wireshark-exported CSV files.

Aggregates packet-level rows (Time, Source, Destination, Protocol, Length, Info)
into bidirectional 5-tuple flows with the same feature schema used by
heuristic_threat_model.py — matching what pcap_to_flows.py produces from raw PCAPs.

Used to convert the Midterm_53_group.csv normal traffic into flow-level features
so it can be evaluated by the heuristic threshold rules.
"""

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── Port → service ────────────────────────────────────────────────────────────
PORT_SERVICE = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet",
    25: "smtp", 53: "dns", 67: "dhcp", 68: "dhcp",
    80: "http", 110: "pop3", 143: "imap", 161: "snmp",
    443: "ssl", 445: "microsoft-ds", 993: "imaps", 995: "pop3s",
    1433: "mssql", 3306: "mysql", 3389: "rdp",
    8080: "http", 8443: "ssl",
}

PROTO_MAP = {
    "TCP": 6, "TLSv1.3": 6, "TLSv1.2": 6, "TLSv1": 6, "SSLv2": 6,
    "HTTP": 6, "OCSP": 6, "STUN": 6,
    "UDP": 17, "DNS": 17, "DHCP": 17, "NBNS": 17,
    "ICMP": 1,
}

FLAG_MAP = {
    "SYN, ACK": 0x12, "PSH, ACK": 0x18, "FIN, ACK": 0x11,
    "RST, ACK": 0x14, "SYN": 0x02, "ACK": 0x10,
    "FIN": 0x01, "RST": 0x04, "PSH": 0x08,
}

PORT_RE = re.compile(r'(\d+)\s*[>→]\s*(\d+)')
FLAG_RE = re.compile(r'\[([^\]]+)\]')

F_FIN, F_SYN, F_RST, F_ACK = 0x01, 0x02, 0x04, 0x10


def _parse_info(info: str, proto: int, proto_name: str = ""):
    sport = dport = flags = None
    if not isinstance(info, str):
        return sport, dport, flags
    m = PORT_RE.search(info)
    if m:
        sport, dport = int(m.group(1)), int(m.group(2))
    fm = FLAG_RE.search(info)
    if fm:
        flags = FLAG_MAP.get(fm.group(1).strip(), 0)
    # Only assign port 53 for actual DNS traffic — not NBNS/DHCP/other UDP
    if proto == 17 and dport is None and proto_name == "DNS":
        dport = 53
    return sport, dport, flags or 0


@dataclass
class Flow:
    key: tuple
    t_first: float = 0.0
    t_last:  float = 0.0
    s_pkts:  int   = 0
    s_bytes: int   = 0
    s_times: list  = field(default_factory=list)
    d_pkts:  int   = 0
    d_bytes: int   = 0
    d_times: list  = field(default_factory=list)
    t_syn:    Optional[float] = None
    t_synack: Optional[float] = None
    t_ack:    Optional[float] = None
    has_fin: bool = False
    has_rst: bool = False
    has_syn: bool = False
    has_ack: bool = False
    dport:   Optional[int] = None
    proto:   int = 0


def _add_packet(flow: Flow, ts, length, flags, is_src):
    if is_src:
        flow.s_pkts  += 1
        flow.s_bytes += length
        flow.s_times.append(ts)
    else:
        flow.d_pkts  += 1
        flow.d_bytes += length
        flow.d_times.append(ts)
    flow.t_last = max(flow.t_last, ts)
    if flags & F_FIN: flow.has_fin = True
    if flags & F_RST: flow.has_rst = True
    if flags & F_SYN: flow.has_syn = True
    if flags & F_ACK: flow.has_ack = True
    if (flags & F_SYN) and not (flags & F_ACK) and flow.t_syn is None:
        flow.t_syn = ts
    if (flags & F_SYN) and (flags & F_ACK) and flow.t_synack is None:
        flow.t_synack = ts
    if (flags & F_ACK) and not (flags & F_SYN) and flow.t_synack and not flow.t_ack:
        flow.t_ack = ts


def _finalise(flow: Flow) -> dict:
    dur   = max(flow.t_last - flow.t_first, 1e-6)
    smean = (flow.s_bytes / flow.s_pkts) if flow.s_pkts else 0
    dmean = (flow.d_bytes / flow.d_pkts) if flow.d_pkts else 0
    rate  = (flow.s_pkts + flow.d_pkts) / dur
    sload = (flow.s_bytes * 8) / dur
    dload = (flow.d_bytes * 8) / dur

    def _ipkt(times):
        if len(times) < 2: return 0.0
        gaps = [(times[i+1]-times[i])*1000 for i in range(len(times)-1)]
        return sum(gaps)/len(gaps)

    synack = (flow.t_synack - flow.t_syn)    if (flow.t_syn and flow.t_synack) else 0.0
    ackdat = (flow.t_ack    - flow.t_synack) if (flow.t_synack and flow.t_ack) else 0.0

    if flow.has_rst:   state = "RST"
    elif flow.has_fin: state = "FIN"
    elif flow.proto == 6 and flow.has_syn and not flow.has_ack: state = "INT"
    elif flow.proto == 6 and flow.has_syn and flow.has_ack:     state = "CON"
    elif flow.proto == 17: state = "CON"
    else: state = "INT"

    src, dst, sport, dport, proto = flow.key
    return {
        "src_ip":  src,  "dst_ip":  dst,
        "sport":   sport or 0, "dport": dport or 0,
        "proto":   proto,
        "service": PORT_SERVICE.get(flow.dport, "-") if flow.dport else "-",
        "state":   state,
        "dur":     round(dur, 6),
        "spkts":   flow.s_pkts,  "dpkts":  flow.d_pkts,
        "sbytes":  flow.s_bytes, "dbytes": flow.d_bytes,
        "rate":    round(rate, 4),
        "sload":   round(sload, 4), "dload": round(dload, 4),
        "sinpkt":  round(_ipkt(flow.s_times), 4),
        "dinpkt":  round(_ipkt(flow.d_times), 4),
        "smean":   round(smean, 2), "dmean": round(dmean, 2),
        "synack":  round(synack, 6), "ackdat": round(ackdat, 6),
        "tcprtt":  round(synack + ackdat, 6),
        "attack_cat": "Normal",
    }


def csv_packets_to_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a Wireshark packet CSV into bidirectional flows."""
    flows      = {}
    first_dir  = {}

    for _, row in df.iterrows():
        try:
            ts = float(row.get("Time") or 0)
        except (ValueError, TypeError):
            continue

        src = str(row.get("Source") or "").strip()
        dst = str(row.get("Destination") or "").strip()
        if not src or not dst or src == "nan" or dst == "nan":
            continue

        proto_name = str(row.get("Protocol") or "")
        proto = PROTO_MAP.get(proto_name, 0)
        if proto == 0:
            continue

        length = int(float(row.get("Length") or 0))
        sport, dport, flags = _parse_info(str(row.get("Info") or ""), proto, proto_name)

        # normalise bidirectional key
        s_key = (src,  sport or 0)
        d_key = (dst,  dport or 0)
        if s_key <= d_key:
            fwd_key = (src, dst, sport, dport, proto)
        else:
            fwd_key = (dst, src, dport, sport, proto)

        if fwd_key not in flows:
            flows[fwd_key]    = Flow(key=fwd_key, t_first=ts, t_last=ts,
                                     proto=proto, dport=dport)
            first_dir[fwd_key] = (src, sport)

        flow      = flows[fwd_key]
        is_src    = (src == first_dir[fwd_key][0])
        if dport and (flow.dport is None or dport < flow.dport):
            flow.dport = dport
        _add_packet(flow, ts, length, flags or 0, is_src)

    rows = [_finalise(f) for f in flows.values()]
    return pd.DataFrame(rows)
