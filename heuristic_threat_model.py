"""
Heuristic threshold-based threat detection model.

Detects Reconnaissance (T1595) and DoS/DDoS (T1498) from:
  - PCAP files   : python3 heuristic_threat_model.py <file.pcap> [out.csv]
  - UNSW-NB15    : python3 heuristic_threat_model.py
  - Normal CSV   : python3 heuristic_threat_model.py --normal <wireshark_csv> [out.csv]

Two detection layers:
  1. Per-flow rules  – thresholds derived from UNSW-NB15 feature distributions
  2. Cross-flow rules – correlate flows per source IP to catch port scanning
                        and flood campaigns invisible in a single flow

UNSW-NB15 baseline (what the thresholds are calibrated against):
  Recon:  spkts p50=10, sbytes p50=564, smean p50=56, dmean p50=44, dur p50=0.75s
  DoS:    spkts p50=10, sbytes p50=834, smean p50=100, sload p50=23K, rate p50=52 pps

Normal traffic baseline (Midterm_53_group.csv — 393K filtered packets):
  Verified false positive rate: 0.85% on known-clean VMware lab traffic
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "unsw_with_normal.csv"       # UNSW-NB15 + Midterm normal flows
DATA_PATH_UNSW   = "unsw_suspicious_events_cleaned.csv"   # fallback (UNSW only)
OUTPUT_CSV       = "heuristic_predictions.csv"
OUTPUT_JSON      = "heuristic_benchmark_results.json"
NORMAL_OUTPUT_CSV  = "normal_traffic_predictions.csv"
NORMAL_OUTPUT_JSON = "normal_traffic_benchmark_results.json"

# Suspicious protocols/patterns to filter from normal traffic CSVs
NORMAL_FILTER_PROTOCOLS = {"ARP", "RARP", "BROWSER", "ICMPv6"}
NORMAL_FILTER_ICMP_BROADCAST = True   # remove ICMP to 255.255.255.255
NORMAL_FILTER_NBNS_WPAD      = True   # remove NBNS WPAD queries

# ── Class labels ──────────────────────────────────────────────────────────────
NORMAL  = 0
RECON   = 1
DOS     = 2
OTHER   = 3
EXFIL   = 4   # outbound high-volume: internal → external, server responds
TOOLXFR = 5  # inbound high-volume: external → internal, likely tool/module download

CLASS_NAMES = {
    NORMAL:  "Normal",
    RECON:   "Reconnaissance",
    DOS:     "DoS",
    OTHER:   "Other Attack",
    EXFIL:   "Suspicious Outbound (Exfil)",
    TOOLXFR: "Suspicious Inbound (Tool Transfer)",
}

# Severity level per rule name — per project requirements
RULE_SEVERITY = {
    # SYN Flood → High
    "SYN_Flood":          "High",
    "SYN_Flood_Campaign": "High",
    # Port Scan → Medium
    "Port_Scan_Probe":    "Medium",
    "Port_Scan_Campaign": "Medium",
    "Stealth_Scan_SubSecond": "Medium",
    "SYN_Scan_NoResponse":    "Medium",
    "IP_Sweep":               "Medium",
    # DNS Tunneling → Critical
    "DNS_Tunneling":          "Critical",
    # Other rules
    "DNS_Recon_Probe":        "Medium",
    "High_Volume_Flood":      "High",
    "High_Rate_Sustained_Flood": "High",
    "Large_Burst_Short":      "High",
    "UDP_ICMP_Flood":         "High",
    "DDoS_Volume_Campaign":   "High",
}

ATTACK_CAT_TO_CLASS = {
    "Reconnaissance": RECON,
    "DoS":            DOS,
    "Normal":         NORMAL,
    "Exploits":       OTHER,
    "Fuzzers":        OTHER,
    "Generic":        OTHER,
    "Backdoor":       OTHER,
    "Shellcode":      OTHER,
    "Worms":          OTHER,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _safe(row, col, default=0.0):
    val = row.get(col, default)
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f  # NaN check
    except (ValueError, TypeError):
        return default


def _is_private_ip(ip: str) -> bool:
    """Return True if ip is an RFC 1918 private address or loopback."""
    if not ip or not isinstance(ip, str):
        return False
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return (
        a == 10                       # 10.0.0.0/8
        or (a == 172 and 16 <= b <= 31)   # 172.16.0.0/12
        or (a == 192 and b == 168)    # 192.168.0.0/16
        or a == 127                   # loopback
    )


def _apply_direction_context(row: dict, class_id: int) -> tuple:
    """
    After a DoS rule fires, use flow direction to distinguish:
      - EXFIL  (T1041): internal src → external dst, server responds (dpkts > 0)
      - TOOLXFR(T1105): external src → internal dst, large inbound payload
    Returns (new_class_id, new_mitre) — unchanged if no reclassification applies.
    """
    if class_id != DOS:
        return class_id, None

    src = str(row.get("src_ip", ""))
    dst = str(row.get("dst_ip", ""))
    src_private = _is_private_ip(src)
    dst_private = _is_private_ip(dst)

    spkts  = _safe(row, "spkts")
    dpkts  = _safe(row, "dpkts")
    sbytes = _safe(row, "sbytes")
    dbytes = _safe(row, "dbytes")

    # Internal → external with server responding: data leaving the network
    if src_private and not dst_private and dpkts > 0:
        return EXFIL, "T1041 — Exfiltration Over C2 Channel"

    # External → internal with large inbound relative to outbound: download/tool transfer
    if not src_private and dst_private and dbytes > sbytes * 2 and dbytes > 5000:
        return TOOLXFR, "T1105 — Ingress Tool Transfer"

    return class_id, None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — PER-FLOW RULES
# Evaluated on every individual flow row.
# Priority order: DoS → Recon → Normal → Other.
# ─────────────────────────────────────────────────────────────────────────────

RECON_RULES = [
    {
        "name": "Port_Scan_Probe",
        "description": (
            "Compact flow with server responding — matches UNSW Recon signature. "
            "UNSW: 85% of Recon rows have spkts==10 (Nmap TCP connect), dpkts median=8, "
            "sbytes p90=900, smean p50=56 (below DoS p50=100). "
            "smean<=90 ceiling separates from DoS (p50=100) and Fuzzers (p50=90). "
            "Excludes well-known service ports (443, 80, 22) to avoid misclassifying "
            "legitimate HTTPS/HTTP/SSH handshakes as port scans."
        ),
        "check": lambda r: (
            _safe(r, "spkts") >= 8
            and _safe(r, "spkts") <= 15
            and _safe(r, "sbytes") <= 800
            and _safe(r, "dbytes") <= 500
            and _safe(r, "dur") < 2.0
            and str(r.get("service", "")).lower() not in ("ssl", "http", "ssh")
            and int(_safe(r, "dport")) not in (443, 80, 22, 8080, 8443)
        ),
        "mitre": "T1595.002 — Vulnerability Scanning",
    },
    {
        "name": "Stealth_Scan_SubSecond",
        "description": (
            "Sub-second flow with symmetric small packets — SYN sweep or ping sweep. "
            "Recon smean p50=56, p90=100; Fuzzers smean p50=90. "
            "smean 40-90 catches most Recon probes while excluding Fuzzers. "
            "Excludes well-known service ports (443, 80, 22) to prevent normal "
            "HTTPS/HTTP/SSH handshakes from being misidentified as stealth scans."
        ),
        "check": lambda r: (
            _safe(r, "dur") < 1.5
            and _safe(r, "spkts") >= 8
            and _safe(r, "smean") >= 40
            and _safe(r, "smean") <= 90
            and _safe(r, "dbytes") <= 400
            and str(r.get("service", "")).lower() not in ("ssl", "http", "ssh")
            and int(_safe(r, "dport")) not in (443, 80, 22, 8080, 8443)
        ),
        "mitre": "T1595.001 — Scanning IP Blocks",
    },
    {
        "name": "DNS_Recon_Probe",
        "description": (
            "DNS-service flow with small payload — DNS zone transfer or host enumeration. "
            "Excludes broadcast source IPs (x.x.x.255) to avoid flagging NBNS/NetBIOS "
            "broadcast traffic which shares UDP port 53 features but is normal Windows "
            "name resolution behaviour."
        ),
        "check": lambda r: (
            str(r.get("service", "")).lower() == "dns"
            and _safe(r, "spkts") >= 3
            and _safe(r, "sbytes") <= 1000
            and not str(r.get("src_ip", "")).endswith(".255")
            and not str(r.get("dst_ip", "")).endswith(".255")
        ),
        "mitre": "T1590.002 — DNS",
    },
    {
        "name": "DNS_Tunneling",
        "description": (
            "DNS flow with unusually large payload — data exfiltration or C2 over DNS. "
            "Legitimate DNS queries are tiny (<512 bytes); tunneling tools (iodine, dnscat2) "
            "encode data in subdomains, producing large DNS flows with many packets. "
            "sbytes > 5000 or smean > 200 catches encoding overhead. "
            "Severity: Critical — bypasses firewalls that allow outbound DNS."
        ),
        "check": lambda r: (
            str(r.get("service", "")).lower() == "dns"
            and (
                _safe(r, "sbytes") > 5000
                or _safe(r, "smean") > 200
                or _safe(r, "spkts") > 20
            )
            and not str(r.get("src_ip", "")).endswith(".255")
        ),
        "mitre": "T1071.004 — DNS (Application Layer Protocol)",
    },
    {
        "name": "SYN_Scan_NoResponse",
        "description": (
            "1-3 src packets, zero dst response, tiny payload — SYN scan or filtered port probe. "
            "In PCAP flows: a SYN with no SYN-ACK means port is filtered or closed. "
            "Kept separate from DoS SYN_Flood (which requires spkts>8 and sload>8K)."
        ),
        "check": lambda r: (
            _safe(r, "spkts") >= 1
            and _safe(r, "spkts") <= 3
            and _safe(r, "dpkts") == 0
            and _safe(r, "sbytes") <= 250          # single SYN or probe
            and _safe(r, "sload") <= 8000          # low load — not a flood
            and _safe(r, "dur") < 1.0
        ),
        "mitre": "T1595.001 — Scanning IP Blocks",
    },
]

DOS_RULES = [
    {
        "name": "SYN_Flood",
        "description": (
            "Many packets from source, server returns nothing — SYN flood. "
            "UNSW: 29.6% of DoS rows have dpkts==0 (vs 5.8% of Exploits). "
            "Requires sload>8K to exclude idle/dead flows and spkts>8 to exclude single probes."
        ),
        "check": lambda r: (
            _safe(r, "spkts") > 8
            and _safe(r, "dpkts") == 0
            and _safe(r, "sload") > 8000
            and (_safe(r, "spkts") + _safe(r, "dpkts")) > 2
        ),
        "mitre": "T1498.001 — Direct Network Flood",
    },
    {
        "name": "UDP_ICMP_Flood",
        "description": (
            "High-volume UDP or ICMP flow — UDP/ICMP amplification or reflection. "
            "Requires multiple packets and real duration to exclude single-packet noise."
        ),
        "check": lambda r: (
            _safe(r, "proto") in [1, 17]
            and _safe(r, "spkts") > 10
            and _safe(r, "sload") > 100000
            and _safe(r, "dur") > 0.0001
        ),
        "mitre": "T1498.002 — Reflection Amplification",
    },
    {
        "name": "High_Volume_Flood",
        "description": (
            "Large total bytes, many packets, source-dominated burst — volumetric TCP flood. "
            "UNSW DoS: sbytes p75=2174, spkts p75=18. "
            "Requires source to send more packets than it receives (spkts > dpkts) "
            "to exclude downloads. Excludes flows where server returns >3x source bytes. "
            "dur < 60s restricts to burst floods — sustained long sessions (streaming, "
            "backups) are not DoS floods and are handled by High_Rate_Sustained_Flood."
        ),
        "check": lambda r: (
            (
                _safe(r, "sbytes") > 5000
                or _safe(r, "dbytes") > 5000
            )
            and _safe(r, "spkts") > 18
            and _safe(r, "spkts") > _safe(r, "dpkts")
            and _safe(r, "dbytes") <= _safe(r, "sbytes") * 3
            and _safe(r, "dur") < 60.0
        ),
        "mitre": "T1498 — Network Denial of Service",
    },
    {
        "name": "High_Rate_Sustained_Flood",
        "description": (
            "Extreme source load sustained over time — amplification or high-rate flood. "
            "UNSW DoS sload p75=1.6M vs Exploits p75=43K. "
            "Requires >1 packet to exclude dead flows."
        ),
        "check": lambda r: (
            _safe(r, "sload") > 500000
            and _safe(r, "dur") > 0
            and (_safe(r, "spkts") + _safe(r, "dpkts")) > 1
        ),
        "mitre": "T1498 — Network Denial of Service",
    },
    {
        "name": "Large_Burst_Short",
        "description": (
            "Large mean packet size, many source packets, sub-second — DoS burst. "
            "UNSW DoS: smean p50=100 vs Recon p50=56. smean>120 above Recon range (p90=100). "
            "Requires spkts > dpkts: real bursts are one-sided. Excludes CDN/HTTPS flows "
            "where the server responds with equal or more packets (jumbo frames, TLS data)."
        ),
        "check": lambda r: (
            _safe(r, "smean") > 120
            and _safe(r, "spkts") > 10
            and _safe(r, "dur") < 1.0
            and _safe(r, "spkts") > _safe(r, "dpkts")
        ),
        "mitre": "T1498 — Network Denial of Service",
    },
]

NORMAL_RULES = [
    {
        "name": "Low_Activity_Flow",
        "description": "Very low traffic volume consistent with benign background activity.",
        "check": lambda r: (
            _safe(r, "spkts") <= 5
            and _safe(r, "dpkts") <= 5
            and _safe(r, "sbytes") <= 300
            and _safe(r, "dur") < 0.5
            and _safe(r, "rate") < 100
        ),
        "mitre": None,
    },
]


def classify_flow(row: dict, pcap_mode: bool = False) -> tuple:
    """
    Returns (class_id, rule_name, mitre_technique).
    Priority: DNS Tunneling → DoS → Recon → Normal → Other.
    DNS Tunneling is checked first to prevent high-volume DNS flows
    from being misclassified as DoS floods.

    In pcap_mode, DoS hits are further refined by direction context:
      - internal→external with server response  → EXFIL  (T1041)
      - external→internal with large inbound    → TOOLXFR (T1105)
    """
    # DNS Tunneling checked before DoS — high-volume DNS is exfil/C2, not a flood
    for rule in RECON_RULES:
        if rule["name"] == "DNS_Tunneling" and rule["check"](row):
            return RECON, rule["name"], rule["mitre"]

    for rule in DOS_RULES:
        if rule["check"](row):
            if pcap_mode:
                new_class, new_mitre = _apply_direction_context(row, DOS)
                if new_class != DOS:
                    return new_class, rule["name"], new_mitre
            return DOS, rule["name"], rule["mitre"]

    for rule in RECON_RULES:
        if rule["check"](row):
            return RECON, rule["name"], rule["mitre"]

    for rule in NORMAL_RULES:
        if rule["check"](row):
            return NORMAL, rule["name"], rule["mitre"]

    return OTHER, "No_Rule_Matched", None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — CROSS-FLOW CORRELATION RULES (PCAP mode only)
#
# These rules look across ALL flows from the same source IP to detect
# attack campaigns that are invisible in any single flow.
# ─────────────────────────────────────────────────────────────────────────────

# Minimum distinct destination ports a single src_ip must probe in small flows
# before being flagged as a port scanner.
PORTSCAN_MIN_PORTS = 10

# Minimum distinct destination IPs a src must contact in small flows (IP sweep)
SWEEP_MIN_HOSTS = 8

# Minimum total bytes a src must send to ONE target to trigger DDoS cross-flow rule.
# Set to 100MB — modern CDN/web traffic easily exceeds 5MB from a single server.
# A real DDoS campaign would generate hundreds of MB in a short capture window.
DDOS_MIN_TOTAL_BYTES = 100_000_000   # 100 MB

# Minimum number of SYN-only flows (dpkts==0) a src must have toward one target
SYN_FLOOD_MIN_FLOWS = 15


def apply_cross_flow_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second-pass analysis over the full flow table.
    Upgrades 'Other Attack' or 'Normal' rows to Recon/DoS where the
    cross-flow pattern is unambiguous.
    """
    df = df.copy()
    df["cross_flow_rule"] = ""

    # Only analyse flows that have valid IP addresses
    has_ip = df["src_ip"].notna() & (df["src_ip"] != "")

    # ── Port Scan Detection ───────────────────────────────────────────────────
    # Flows eligible for scan counting: small, short, few packets.
    # dport < 10000 restricts to server/service ports — excludes ephemeral client
    # ports (typically 32768-60999) that appear as "dport" after bidirectional
    # flow normalisation of HTTP/HTTPS browsing traffic.
    scan_eligible = (
        has_ip
        & (df["spkts"] <= 15)
        & (df["sbytes"] <= 1200)
        & (df["dur"] < 3.0)
        & (df["dport"] < 10000)     # server ports only — not random client ephemeral ports
    )

    # Count distinct dst_ports each src_ip probed in scan-eligible flows
    scan_flows   = df[scan_eligible]
    ports_by_src = (
        scan_flows.groupby("src_ip")["dport"]
        .nunique()
        .rename("distinct_ports")
    )
    scanning_srcs = set(ports_by_src[ports_by_src >= PORTSCAN_MIN_PORTS].index)

    # Flag all scan-eligible flows from scanning source IPs
    scan_mask = scan_eligible & df["src_ip"].isin(scanning_srcs)
    df.loc[scan_mask, "pred_class"]     = RECON
    df.loc[scan_mask, "pred_label"]     = CLASS_NAMES[RECON]
    df.loc[scan_mask, "rule_fired"]     = "Port_Scan_Campaign"
    df.loc[scan_mask, "mitre_technique"] = "T1595 — Active Scanning"
    df.loc[scan_mask, "cross_flow_rule"] = f"src probed ≥{PORTSCAN_MIN_PORTS} distinct ports"

    # ── IP Sweep Detection ────────────────────────────────────────────────────
    # Same source contacting many distinct destination IPs with tiny payloads.
    # dport < 10000 same reasoning as port scan above.
    sweep_eligible = (
        has_ip
        & (df["spkts"] <= 5)
        & (df["sbytes"] <= 500)
        & (df["dur"] < 1.0)
        & (df["dport"] < 10000)
    )
    sweep_flows   = df[sweep_eligible]
    hosts_by_src  = (
        sweep_flows.groupby("src_ip")["dst_ip"]
        .nunique()
        .rename("distinct_hosts")
    )
    sweeping_srcs = set(hosts_by_src[hosts_by_src >= SWEEP_MIN_HOSTS].index)

    sweep_mask = sweep_eligible & df["src_ip"].isin(sweeping_srcs)
    # Only upgrade rows not already flagged as Recon
    upgrade_sweep = sweep_mask & (df["pred_class"] != RECON)
    df.loc[upgrade_sweep, "pred_class"]      = RECON
    df.loc[upgrade_sweep, "pred_label"]      = CLASS_NAMES[RECON]
    df.loc[upgrade_sweep, "rule_fired"]      = "IP_Sweep"
    df.loc[upgrade_sweep, "mitre_technique"] = "T1595.001 — Scanning IP Blocks"
    df.loc[upgrade_sweep, "cross_flow_rule"] = f"src contacted ≥{SWEEP_MIN_HOSTS} distinct hosts"

    # ── DDoS Campaign Detection ───────────────────────────────────────────────
    # Source IP sends very high total bytes to one specific target
    pair_bytes = (
        df.groupby(["src_ip", "dst_ip"])["sbytes"]
        .sum()
        .rename("total_sbytes")
        .reset_index()
    )
    flooder_pairs = pair_bytes[pair_bytes["total_sbytes"] >= DDOS_MIN_TOTAL_BYTES]
    flooder_set   = set(zip(flooder_pairs["src_ip"], flooder_pairs["dst_ip"]))

    ddos_pair_mask = has_ip & df.apply(
        lambda r: (r["src_ip"], r["dst_ip"]) in flooder_set, axis=1
    )
    upgrade_ddos = ddos_pair_mask & ~df["pred_class"].isin([DOS])
    df.loc[upgrade_ddos, "pred_class"]      = DOS
    df.loc[upgrade_ddos, "pred_label"]      = CLASS_NAMES[DOS]
    df.loc[upgrade_ddos, "rule_fired"]      = "DDoS_Volume_Campaign"
    df.loc[upgrade_ddos, "mitre_technique"] = "T1498 — Network Denial of Service"
    df.loc[upgrade_ddos, "cross_flow_rule"] = f"pair total sbytes ≥{DDOS_MIN_TOTAL_BYTES:,}"

    # ── SYN Flood Campaign ────────────────────────────────────────────────────
    # Source hammers one target with many half-open (dpkts==0) flows.
    # dport < 10000 restricts to server/service ports — excludes cases where a
    # server (e.g. Google) opens connections to client ephemeral ports (push/WebRTC)
    # which produces many half-open flows but is not a SYN flood.
    syn_flows     = df[has_ip & (df["dpkts"] == 0) & (df["spkts"] > 1) & (df["dport"] < 10000)]
    syn_per_pair  = syn_flows.groupby(["src_ip", "dst_ip"]).size().rename("syn_flow_count")
    syn_pairs     = set(zip(
        syn_per_pair[syn_per_pair >= SYN_FLOOD_MIN_FLOWS].index.get_level_values(0),
        syn_per_pair[syn_per_pair >= SYN_FLOOD_MIN_FLOWS].index.get_level_values(1),
    ))

    syn_flood_mask = has_ip & (df["dpkts"] == 0) & df.apply(
        lambda r: (r["src_ip"], r["dst_ip"]) in syn_pairs, axis=1
    )
    upgrade_syn = syn_flood_mask & ~df["pred_class"].isin([DOS])
    df.loc[upgrade_syn, "pred_class"]      = DOS
    df.loc[upgrade_syn, "pred_label"]      = CLASS_NAMES[DOS]
    df.loc[upgrade_syn, "rule_fired"]      = "SYN_Flood_Campaign"
    df.loc[upgrade_syn, "mitre_technique"] = "T1498.001 — Direct Network Flood"
    df.loc[upgrade_syn, "cross_flow_rule"] = f"≥{SYN_FLOOD_MIN_FLOWS} half-open flows to same target"

    # ── SIEM Correlation Rule — Potential Intrusion Attempt ───────────────────
    # Port Scan + Failed Logins + PowerShell Download from same src_ip
    # Port scan: src already flagged as Recon (port scan/sweep rules)
    # Failed login: short TCP flow to auth ports (22/23/3389/445/21) with dpkts==0
    # PowerShell/tool download: inbound large payload (TOOLXFR) to same or related host
    AUTH_PORTS = {22, 23, 21, 3389, 445, 1433, 3306}
    failed_login_mask = (
        has_ip
        & (df["proto"] == 6)
        & df["dport"].isin(AUTH_PORTS)
        & (df["dpkts"] == 0)
        & (df["spkts"] <= 3)
    )
    failed_login_srcs = set(df[failed_login_mask]["src_ip"].dropna())

    toolxfr_mask = df["pred_class"] == TOOLXFR
    toolxfr_dsts = set(df[toolxfr_mask]["dst_ip"].dropna())

    recon_srcs = set(df[df["pred_class"] == RECON]["src_ip"].dropna())

    # A src that port-scanned AND had failed auth attempts qualifies;
    # if any internal host also received a tool download → full chain
    intrusion_srcs = recon_srcs & failed_login_srcs
    if intrusion_srcs and toolxfr_dsts:
        intrusion_mask = (
            has_ip
            & df["src_ip"].isin(intrusion_srcs)
        )
        df.loc[intrusion_mask, "siem_alert"] = "Potential Intrusion Attempt"
    elif intrusion_srcs:
        intrusion_mask = has_ip & df["src_ip"].isin(intrusion_srcs)
        df.loc[intrusion_mask, "siem_alert"] = "Potential Intrusion Attempt (no download observed)"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# APPLY RULES TO DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def run_on_dataframe(df: pd.DataFrame, pcap_mode: bool = False) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        pred_class, rule_name, mitre = classify_flow(row.to_dict(), pcap_mode=pcap_mode)
        results.append({
            "pred_class":      pred_class,
            "pred_label":      CLASS_NAMES[pred_class],
            "rule_fired":      rule_name,
            "mitre_technique": mitre or "",
            "severity":        RULE_SEVERITY.get(rule_name, "Low"),
        })

    pred_df = pd.DataFrame(results)
    df = df.copy()
    df["pred_class"]      = pred_df["pred_class"].values
    df["pred_label"]      = pred_df["pred_label"].values
    df["rule_fired"]      = pred_df["rule_fired"].values
    df["mitre_technique"] = pred_df["mitre_technique"].values
    df["severity"]        = pred_df["severity"].values
    df["siem_alert"]      = ""

    if pcap_mode:
        df["cross_flow_rule"] = ""
        if "src_ip" in df.columns:
            df = apply_cross_flow_rules(df)
    else:
        if "attack_cat" in df.columns:
            df["true_class"] = (
                df["attack_cat"].map(ATTACK_CAT_TO_CLASS).fillna(OTHER).astype(int)
            )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_pcap_report(df: pd.DataFrame, source_name: str):
    threat_classes = [RECON, DOS, EXFIL, TOOLXFR]
    flagged = df[df["pred_class"].isin(threat_classes)]
    total   = len(df)

    print(f"\n{'='*65}")
    print(f"  THREAT HUNTING REPORT — {os.path.basename(source_name)}")
    print(f"{'='*65}")
    print(f"  Total flows analysed : {total:,}")
    print(f"  Threats detected     : {len(flagged):,}  ({len(flagged)/total*100:.1f}%)")

    print(f"\n── Class breakdown ──")
    threat_labels = {
        "Reconnaissance":              "← THREAT",
        "DoS":                         "← THREAT",
        "Suspicious Outbound (Exfil)": "← THREAT  T1041",
        "Suspicious Inbound (Tool Transfer)": "← THREAT  T1105",
    }
    for label in ["Reconnaissance", "DoS",
                  "Suspicious Outbound (Exfil)", "Suspicious Inbound (Tool Transfer)",
                  "Normal", "Other Attack"]:
        count = (df["pred_label"] == label).sum()
        pct   = count / total * 100
        tag   = f"  {threat_labels[label]}" if label in threat_labels else ""
        print(f"  {label:<38s} {count:>6,}  ({pct:5.1f}%){tag}")

    if len(flagged) == 0:
        print("\n  No threats detected.")
        return

    print(f"\n── Rules triggered (Layer 1 — per-flow) ──")
    layer1 = flagged[flagged["cross_flow_rule"] == ""]
    for rule, cnt in layer1["rule_fired"].value_counts().items():
        print(f"  {rule:<45s} {cnt:>5,}")

    cross = flagged[flagged["cross_flow_rule"] != ""]
    if len(cross):
        print(f"\n── Rules triggered (Layer 2 — cross-flow correlation) ──")
        for rule, cnt in cross["rule_fired"].value_counts().items():
            note = cross[cross["rule_fired"] == rule]["cross_flow_rule"].iloc[0]
            print(f"  {rule:<35s} {cnt:>5,}  [{note}]")

    print(f"\n── Severity breakdown ──")
    sev_order = ["Critical", "High", "Medium", "Low"]
    for sev in sev_order:
        cnt = (flagged["severity"] == sev).sum()
        if cnt:
            print(f"  {sev:<10s} {cnt:>6,}")

    print(f"\n── MITRE ATT&CK techniques ──")
    for tech, cnt in flagged["mitre_technique"].value_counts().items():
        if tech:
            print(f"  {tech:<50s} {cnt:>5,}")

    # SIEM alerts
    if "siem_alert" in df.columns:
        siem = df[df["siem_alert"] != ""]
        if len(siem):
            print(f"\n{'!'*65}")
            print(f"  !! SIEM ALERT — CORRELATION RULE TRIGGERED !!")
            print(f"{'!'*65}")
            for alert, cnt in siem["siem_alert"].value_counts().items():
                print(f"  {alert}  ({cnt:,} flows affected)")
            alert_srcs = siem["src_ip"].value_counts().head(5) if "src_ip" in siem.columns else []
            if len(alert_srcs):
                print(f"  Source IPs involved:")
                for ip, cnt in alert_srcs.items():
                    print(f"    {ip}  ({cnt:,} flows)")

    # Top Recon — highest distinct port count sources
    recon_flows = df[df["pred_class"] == RECON]
    if len(recon_flows):
        print(f"\n── Top Reconnaissance sources ──")
        recon_cols = [c for c in ["src_ip","dst_ip","sport","dport","spkts","sbytes","smean","dur","rule_fired"] if c in df.columns]
        if "src_ip" in recon_flows.columns:
            src_summary = (
                recon_flows.groupby("src_ip")
                .agg(flows=("pred_class","count"), ports=("dport","nunique"), total_bytes=("sbytes","sum"))
                .sort_values("flows", ascending=False)
                .head(5)
            )
            print(src_summary.to_string())
        else:
            print(recon_flows[recon_cols].head(5).to_string(index=False))

    # Top DoS — highest sload flows
    dos_flows = df[df["pred_class"] == DOS]
    if len(dos_flows):
        print(f"\n── Top DoS flows by source load ──")
        dos_cols = [c for c in ["src_ip","dst_ip","dport","proto","dur","spkts","dpkts","sbytes","sload","rule_fired"] if c in df.columns]
        print(dos_flows.sort_values("sload", ascending=False).head(8)[dos_cols].to_string(index=False))

    # Exfiltration flows — internal → external with server response
    exfil_flows = df[df["pred_class"] == EXFIL]
    if len(exfil_flows):
        print(f"\n── Suspicious Outbound / Exfiltration flows (T1041) ──")
        print(f"  {len(exfil_flows):,} flow(s) from internal hosts to external destinations")
        exfil_cols = [c for c in ["src_ip","dst_ip","dport","proto","dur","spkts","dpkts","sbytes","dbytes","rule_fired"] if c in df.columns]
        print(exfil_flows.sort_values("sbytes", ascending=False).head(8)[exfil_cols].to_string(index=False))

    # Tool transfer flows — external → internal large inbound
    toolxfr_flows = df[df["pred_class"] == TOOLXFR]
    if len(toolxfr_flows):
        print(f"\n── Suspicious Inbound / Tool Transfer flows (T1105) ──")
        print(f"  {len(toolxfr_flows):,} flow(s) from external hosts delivering large payloads inbound")
        txfr_cols = [c for c in ["src_ip","dst_ip","dport","proto","dur","spkts","dpkts","sbytes","dbytes","rule_fired"] if c in df.columns]
        print(toolxfr_flows.sort_values("dbytes", ascending=False).head(8)[txfr_cols].to_string(index=False))


def filter_normal_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known-suspicious packets from a Wireshark CSV before aggregation."""
    original = len(df)

    if NORMAL_FILTER_ICMP_BROADCAST:
        df = df[~((df["Protocol"] == "ICMP") & (df["Destination"] == "255.255.255.255"))]

    if NORMAL_FILTER_NBNS_WPAD:
        df = df[~((df["Protocol"] == "NBNS") & (df["Info"].str.contains("WPAD", na=False)))]

    df = df[~df["Protocol"].isin(NORMAL_FILTER_PROTOCOLS)]
    df = df.reset_index(drop=True)

    removed = original - len(df)
    print(f"  Filtered {removed:,} suspicious packets → {len(df):,} clean packets kept")
    return df


def print_normal_report(df: pd.DataFrame, source_name: str,
                        flows_df: pd.DataFrame) -> dict:
    """Report false positive analysis on known-clean normal traffic."""
    total  = len(flows_df)
    flagged = flows_df[flows_df["pred_class"].isin([RECON, DOS])]
    normal  = flows_df[flows_df["pred_class"] == NORMAL]
    other   = flows_df[flows_df["pred_class"] == OTHER]

    fpr = len(flagged) / total * 100 if total else 0

    print(f"\n{'='*65}")
    print(f"  NORMAL TRAFFIC ANALYSIS — {os.path.basename(source_name)}")
    print(f"{'='*65}")
    print(f"  Total flows analysed    : {total:,}")
    print(f"  Correctly benign        : {len(normal):,}  ({len(normal)/total*100:.1f}%)")
    print(f"  Classified Other        : {len(other):,}  ({len(other)/total*100:.1f}%)")
    print(f"  FALSE POSITIVES (threats flagged on clean traffic):")
    print(f"    Recon  : {(flows_df['pred_class']==RECON).sum():,}")
    print(f"    DoS    : {(flows_df['pred_class']==DOS).sum():,}")
    print(f"    Total  : {len(flagged):,}  → False Positive Rate = {fpr:.2f}%")
    print(f"  Target FPR              : <5%  →  {'✓ PASS' if fpr < 5 else '✗ FAIL'}")

    if len(flagged):
        print(f"\n── False positive breakdown by rule ──")
        for rule, cnt in flagged["rule_fired"].value_counts().items():
            print(f"  {rule:<45s} {cnt:>5,}")

        print(f"\n── Sample false positive flows ──")
        fp_cols = [c for c in ["src_ip","dst_ip","dport","proto","dur",
                                "spkts","dpkts","sbytes","sload","smean",
                                "rule_fired"] if c in flows_df.columns]
        print(flagged.head(8)[fp_cols].to_string(index=False))

    # Protocol breakdown of flagged flows
    if len(flagged) and "proto" in flows_df.columns:
        print(f"\n── False positives by protocol ──")
        proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
        for proto, cnt in flagged["proto"].value_counts().items():
            print(f"  proto {proto} ({proto_map.get(int(proto),'?'):<5}) : {cnt:,}")

    return {
        "source": source_name,
        "total_flows": int(total),
        "correctly_benign": int(len(normal)),
        "false_positives": int(len(flagged)),
        "false_positive_rate_pct": round(fpr, 4),
        "target_fpr_pct": 5.0,
        "passes_target": fpr < 5.0,
        "fp_by_rule": flagged["rule_fired"].value_counts().to_dict() if len(flagged) else {},
    }


def print_unsw_report(df: pd.DataFrame):
    rule_counts = df["rule_fired"].value_counts()
    pred_dist   = df["pred_label"].value_counts()

    print("\n── Rule firing counts ──")
    for rule, count in rule_counts.items():
        print(f"  {rule:<45s} {count:>7,}  ({count/len(df)*100:.1f}%)")

    print("\n── Predicted class distribution ──")
    for cls, count in pred_dist.items():
        print(f"  {cls:<20s} {count:>7,}")

    # ── Detection performance on attack traffic ───────────────────────────────
    eval_df = df[df["attack_cat"].isin(["Reconnaissance", "DoS"])].copy()
    print(f"\n── Detection performance (Recon + DoS, n={len(eval_df):,}) ──")
    for attack, class_id in [("Reconnaissance", RECON), ("DoS", DOS)]:
        subset = eval_df[eval_df["attack_cat"] == attack]
        caught = (subset["pred_class"] == class_id).sum()
        total  = len(subset)
        print(f"\n  {attack} (n={total:,})")
        print(f"    Detected : {caught:,}  ({caught/total*100:.1f}%)")
        print(f"    Missed   : {total-caught:,}  ({(total-caught)/total*100:.1f}%)")
        missed = subset[subset["pred_class"] != class_id]
        if len(missed):
            for lbl, cnt in missed["pred_label"].value_counts().items():
                print(f"      → classified as {lbl}: {cnt}")

    # ── False positive rate on real normal traffic ────────────────────────────
    normal_df = df[df["attack_cat"] == "Normal"]
    if len(normal_df):
        fp_on_normal = normal_df[normal_df["pred_class"].isin([RECON, DOS])]
        fpr = len(fp_on_normal) / len(normal_df) * 100
        print(f"\n── False positive rate on normal traffic (n={len(normal_df):,}) ──")
        print(f"  Correctly benign : {(normal_df['pred_class'] == NORMAL).sum():,}  "
              f"({(normal_df['pred_class'] == NORMAL).sum()/len(normal_df)*100:.1f}%)")
        print(f"  False positives  : {len(fp_on_normal):,}  ({fpr:.2f}%)")
        print(f"  Target FPR       : <5%  →  {'✓ PASS' if fpr < 5 else '✗ FAIL'}")
        if len(fp_on_normal):
            print(f"  FP by rule:")
            for rule, cnt in fp_on_normal["rule_fired"].value_counts().items():
                print(f"    {rule:<45s} {cnt:>5,}")

    print(f"\n── Classification report ──")
    print(classification_report(
        df["true_class"], df["pred_class"],
        labels=[NORMAL, RECON, DOS, OTHER],
        target_names=["Normal", "Reconnaissance", "DoS", "Other Attack"],
        zero_division=0,
    ))

    fp_recon = df[(df["true_class"] == OTHER) & (df["pred_class"] == RECON)]
    fp_dos   = df[(df["true_class"] == OTHER) & (df["pred_class"] == DOS)]
    print(f"False positives — OTHER flagged as Recon : {len(fp_recon):,}")
    print(f"False positives — OTHER flagged as DoS   : {len(fp_dos):,}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    normal_mode = len(sys.argv) > 1 and sys.argv[1] == "--normal"
    pcap_mode   = (not normal_mode) and len(sys.argv) > 1 and sys.argv[1].endswith(".pcap")
    pcap_path   = sys.argv[1] if pcap_mode else None
    custom_out  = sys.argv[2] if (pcap_mode and len(sys.argv) > 2) else None

    # ── Normal traffic mode ───────────────────────────────────────────────────
    if normal_mode:
        if len(sys.argv) < 3:
            print("Usage: python3 heuristic_threat_model.py --normal <wireshark_csv> [out.csv]")
            sys.exit(1)

        from csv_to_flows import csv_packets_to_flows

        normal_csv = sys.argv[2]
        out_csv    = sys.argv[3] if len(sys.argv) > 3 else NORMAL_OUTPUT_CSV

        print(f"[1/4] Loading normal traffic CSV: {normal_csv}")
        raw = pd.read_csv(normal_csv)
        print(f"      {len(raw):,} packets loaded")

        print(f"[2/4] Filtering suspicious packets ...")
        clean = filter_normal_csv(raw)

        print(f"[3/4] Aggregating packets → flows ...")
        flows_df = csv_packets_to_flows(clean)
        print(f"      {len(flows_df):,} flows extracted")

        if flows_df.empty:
            print("No flows extracted.")
            return

        print(f"[4/4] Applying heuristic rules (Layer 1 + Layer 2) ...")
        flows_df = run_on_dataframe(flows_df, pcap_mode=True)

        results = print_normal_report(flows_df, normal_csv, flows_df)

        flows_df.to_csv(out_csv, index=False)
        print(f"\nDetailed results saved → {out_csv}")

        with open(NORMAL_OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark saved        → {NORMAL_OUTPUT_JSON}")
        return

    # ── PCAP mode ─────────────────────────────────────────────────────────────
    if pcap_mode:
        from pcap_to_flows import pcap_to_flows

        print(f"[1/3] Converting PCAP → flows: {pcap_path}")
        df = pcap_to_flows(pcap_path)

        if df.empty:
            print("No packets found in PCAP.")
            return

        print(f"      Extracted {len(df):,} flows")

        print(f"[2/3] Applying per-flow rules (Layer 1) + direction context...")
        df = run_on_dataframe(df, pcap_mode=True)

        threat_cls = [RECON, DOS, EXFIL, TOOLXFR]
        l1_threats = df["pred_class"].isin(threat_cls).sum()
        print(f"      Layer 1 flagged {l1_threats:,} threats "
              f"(DoS={( df['pred_class']==DOS).sum()}, "
              f"Exfil={(df['pred_class']==EXFIL).sum()}, "
              f"ToolXfr={(df['pred_class']==TOOLXFR).sum()}, "
              f"Recon={(df['pred_class']==RECON).sum()})")
        print(f"[2/3] Applying cross-flow correlation rules (Layer 2)...")
        l2_threats = df["pred_class"].isin(threat_cls).sum()
        print(f"      Layer 2 upgraded {l2_threats - l1_threats:,} additional flows")

        print_pcap_report(df, pcap_path)

        base    = os.path.splitext(os.path.basename(pcap_path))[0]
        out_csv = custom_out or f"{base}_threats.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nDetailed results saved → {out_csv}")

    else:
        path = DATA_PATH if os.path.exists(DATA_PATH) else DATA_PATH_UNSW
        if path == DATA_PATH_UNSW:
            print(f"[UNSW mode] Note: {DATA_PATH} not found, falling back to {DATA_PATH_UNSW}")
            print(f"  Run: python3 heuristic_threat_model.py --normal <csv> to generate combined dataset")
        print(f"[UNSW mode] Loading {path} ...")
        df = pd.read_csv(path)
        print(f"  {len(df):,} rows  ({df['attack_cat'].value_counts().get('Normal',0):,} normal traffic rows)")

        print("Applying per-flow rules...")
        df = run_on_dataframe(df, pcap_mode=False)

        print_unsw_report(df)

        df.to_csv(OUTPUT_CSV, index=False)
        acc = accuracy_score(df["true_class"], df["pred_class"])
        f1  = f1_score(df["true_class"], df["pred_class"], average="weighted", zero_division=0)
        print(f"\nAccuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}")

        benchmark = {
            "model": "Heuristic Threshold (Rule-Based)",
            "dataset": "UNSW-NB15",
            "total_rows": int(len(df)),
            "overall_accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "per_flow_rules": {
                "recon": [r["name"] for r in RECON_RULES],
                "dos":   [r["name"] for r in DOS_RULES],
            },
            "cross_flow_rules": [
                "Port_Scan_Campaign", "IP_Sweep",
                "DDoS_Volume_Campaign", "SYN_Flood_Campaign"
            ],
            "thresholds": {
                "port_scan_min_ports":    PORTSCAN_MIN_PORTS,
                "ip_sweep_min_hosts":     SWEEP_MIN_HOSTS,
                "ddos_min_total_bytes":   DDOS_MIN_TOTAL_BYTES,
                "syn_flood_min_flows":    SYN_FLOOD_MIN_FLOWS,
            },
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"Benchmark saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
