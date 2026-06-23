import pandas as pd
import numpy as np
import json
from datetime import datetime

# ─── MITRE ATT&CK Rule Definitions ───────────────────────────────────────────

MITRE_RULES = [
    {
        "tactic": "Reconnaissance",
        "technique": "Active Scanning",
        "technique_id": "T1595",
        "severity": "Medium",
        "match": lambda r: (
            r.get("attack_cat") == "Reconnaissance" or
            r.get("attack_cat") == "Fuzzers" or
            (r.get("rate", 0) > 1000 and r.get("sbytes", 0) < 500)
        ),
    },
    {
        "tactic": "Initial Access",
        "technique": "Exploit Public-Facing Application",
        "technique_id": "T1190",
        "severity": "High",
        "match": lambda r: (
            r.get("attack_cat") == "Exploits" or
            r.get("attack_cat") == "Shellcode"
        ),
    },
    {
        "tactic": "Command and Control",
        "technique": "Application Layer Protocol",
        "technique_id": "T1071",
        "severity": "Critical",
        "match": lambda r: (
            r.get("attack_cat") == "Backdoor" or
            (r.get("sinpkt", 0) > 0 and r.get("dinpkt", 0) > 0 and
             abs(r.get("sinpkt", 0) - r.get("dinpkt", 0)) < 0.01 and
             r.get("dur", 0) > 1)
        ),
    },
    {
        "tactic": "Exfiltration",
        "technique": "Exfiltration Over C2 Channel",
        "technique_id": "T1041",
        "severity": "Critical",
        "match": lambda r: (
            r.get("sbytes", 0) > 0 and
            r.get("dbytes", 0) > 0 and
            r.get("sbytes", 0) > r.get("dbytes", 0) * 5
        ),
    },
    {
        "tactic": "Impact",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "High",
        "match": lambda r: (
            r.get("attack_cat") == "DoS" or
            r.get("attack_cat") == "Generic"
        ),
    },
    {
        "tactic": "Lateral Movement",
        "technique": "Remote Services",
        "technique_id": "T1021",
        "severity": "High",
        "match": lambda r: (
            r.get("service") in ["smb", "netbios-ssn", "microsoft-ds"] or
            r.get("attack_cat") == "Worms"
        ),
    },
    {
        "tactic": "Credential Access",
        "technique": "Brute Force",
        "technique_id": "T1110",
        "severity": "Critical",
        "match": lambda r: (
            r.get("is_ftp_login", 0) == 1 or
            r.get("ct_ftp_cmd", 0) > 5
        ),
    },
]

SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# ─── Rule Engine ─────────────────────────────────────────────────────────────

def apply_rules(row):
    matched = []
    for rule in MITRE_RULES:
        try:
            if rule["match"](row):
                matched.append(rule)
        except Exception:
            continue

    if not matched:
        return "Normal", "No Match", "N/A", "Low"

    # Pick highest severity match
    best = max(matched, key=lambda r: SEVERITY_ORDER[r["severity"]])
    tactics = " | ".join(dict.fromkeys(r["tactic"] for r in matched))
    return tactics, best["technique"], best["technique_id"], best["severity"]


def score_severity(row_severity, existing_severity):
    if SEVERITY_ORDER.get(row_severity, 0) > SEVERITY_ORDER.get(existing_severity, 0):
        return row_severity
    return existing_severity


# ─── Load & Process UNSW Data ────────────────────────────────────────────────

def process_unsw(path):
    print(f"[*] Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"    {len(df):,} rows loaded")

    records = df.to_dict(orient="records")
    results = []

    for row in records:
        tactic, technique, technique_id, rule_severity = apply_rules(row)
        final_severity = score_severity(rule_severity, str(row.get("severity", "Low")))
        results.append({
            **row,
            "mitre_tactic": tactic,
            "mitre_technique": technique,
            "mitre_technique_id": technique_id,
            "computed_severity": final_severity,
            "data_source": "UNSW-NB15",
        })

    return pd.DataFrame(results)


# ─── Load & Process DDoS Data ────────────────────────────────────────────────

def process_ddos(path):
    print(f"[*] Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"    {len(df):,} rows loaded")

    df = df[df["target_label"] == 1].copy()
    print(f"    {len(df):,} attack rows kept")

    df["attack_cat"] = "DoS"
    df["mitre_tactic"] = "Impact"
    df["mitre_technique"] = "Network Denial of Service"
    df["mitre_technique_id"] = "T1498"
    df["computed_severity"] = "High"
    df["data_source"] = "DDoS-Dataset"
    df["is_ftp_login"] = 0
    df["service"] = "unknown"

    return df


# ─── Summary Stats ───────────────────────────────────────────────────────────

def generate_summary(df):
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_events": len(df),
        "tactic_breakdown": df["mitre_tactic"].value_counts().to_dict(),
        "severity_breakdown": df["computed_severity"].value_counts().to_dict(),
        "attack_category_breakdown": df["attack_cat"].value_counts().to_dict()
        if "attack_cat" in df.columns else {},
        "top_techniques": df["mitre_technique"].value_counts().head(10).to_dict(),
    }
    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    unsw_df = process_unsw("unsw_suspicious_events_cleaned.csv")
    ddos_df = process_ddos("processed_ddos_data_cleaned.csv")

    # Save separately — no merging to avoid empty columns
    unsw_df.to_csv("threat_hunting_results_unsw.csv", index=False)
    print(f"[+] UNSW results saved to threat_hunting_results_unsw.csv ({len(unsw_df):,} rows)")

    ddos_df.to_csv("threat_hunting_results_ddos.csv", index=False)
    print(f"[+] DDoS results saved to threat_hunting_results_ddos.csv ({len(ddos_df):,} rows)")

    # Save summaries
    unsw_summary = generate_summary(unsw_df)
    ddos_summary = generate_summary(ddos_df)

    with open("threat_hunting_summary_unsw.json", "w") as f:
        json.dump(unsw_summary, f, indent=2)
    with open("threat_hunting_summary_ddos.json", "w") as f:
        json.dump(ddos_summary, f, indent=2)
    print("[+] Summaries saved to threat_hunting_summary_unsw.json and threat_hunting_summary_ddos.json")

    # Print summary to console
    for label, summary in [("UNSW-NB15", unsw_summary), ("DDoS", ddos_summary)]:
        print(f"\n── {label} Threat Hunting Summary ──────────────────────────────")
        print(f"  Total Events : {summary['total_events']:,}")
        print(f"\n  MITRE Tactics:")
        for tactic, count in sorted(summary["tactic_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {tactic:<35} {count:>8,}")
        print(f"\n  Severity:")
        for sev, count in sorted(summary["severity_breakdown"].items(), key=lambda x: -SEVERITY_ORDER.get(x[0], 0)):
            print(f"    {sev:<15} {count:>8,}")


if __name__ == "__main__":
    main()
