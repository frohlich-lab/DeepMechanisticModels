import re


def get_samples(data_name):
    if data_name == "dream_cytof":
        return [
            # Subchallenge 4
            "c184A1",
            "cBT20",  # EGFR amplification (10-14 times)
            "cBT474",
            "cBT549",
            "cCAL148",
            "cCAL851",
            "cCAL51",  # mind that this cell-line DOES NOT follow alphabetical order -- microsatellite instability (MSI)
            "cDU4475",
            "cEFM192A",
            "cEVSAT",
            "cHBL100",
            "cHCC1187",
            "cHCC1395",
            "cHCC1419",
            "cHCC1500",
            "cHCC1569",
            # "cHCC1599",  # outlier
            "cHCC1937",
            "cHCC1954",
            # "cHCC2157", no transcriptomic data
            "cHCC2185",
            "cHCC3153",
            "cHCC38",
            "cHCC70",
            "cHDQP1",
            "cJIMT1",
            "cMCF10A",
            # "cMCF10F", no transcriptomic data
            "cMCF7",
            "cMDAMB134VI",
            "cMDAMB157",
            "cMDAMB175VII",
            "cMDAMB361",
            "cMDAMB415",
            "cMDAMB453",
            # "cMDAkb2", no transcriptomic data
            "cMFM223",
            "cMPE600",
            "cMX1",
            "cOCUBM",
            "cT47D",
            "cUACC812",
            "cUACC893",
            "cZR7530",
            # Subchallenge 2 (added 11.04.2025)
            "c184B5",
            "cBT483",
            "cHCC1428",
            "cHCC1806",
            "cHCC202",
            "cHs578T",
            "cMCF12A",
            "cMDAMB231",
            "cMDAMB468",
            "cSKBR3",
            "cUACC3199",
            "cZR751",
        ]
    elif m := re.match(r"synthetic_([0-9]+)_[0-9.]+_[0-9.]+$", data_name):
        return [f"sample_{isample}" for isample in range(int(m.group(1)))]

    raise ValueError(f"{data_name} is not a valid data name")
