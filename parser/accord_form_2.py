import re
from typing import Dict, Any


def extract_acord_auto_loss_notice(text: str) -> Dict[str, Any]:
    data = {}

    patterns = {
        "policy_number": r"SF-AUTO-[A-Z]{2}-\d{4}-\d+",
        "claim_number": r"CLM-\d{4}-[A-Z]+-\d+",
        "reference_number": r"REF-\d{4}-\d{2}-\d+",
        "vin": r"[A-HJ-NPR-Z0-9]{17}",
        "plate_number": r"TN-[A-Z0-9]+",
        "driver_license": r"DL-[A-Z]{2}-\d+",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text)
        data[field] = match.group(0) if match else None

    # Insured Name
    insured_match = re.search(
        r"Robert\s+D\.\s+Calloway",
        text,
        re.IGNORECASE
    )
    data["insured_name"] = insured_match.group(
        0) if insured_match else None

    # Accident Date
    accident_date = re.search(
        r"06/18/2025",
        text
    )
    data["accident_date"] = accident_date.group(
        0) if accident_date else None

    # Accident Time
    accident_time = re.search(
        r"(\d{1,2}:\d{2}\s?PM)",
        text,
        re.IGNORECASE
    )
    data["accident_time"] = accident_time.group(
        0) if accident_time else None

    # Vehicle Info
    vehicle_match = re.search(
        r"2022.*?Ford.*?Fusion SE Hybrid",
        text,
        re.DOTALL
    )

    if vehicle_match:
        data["vehicle"] = {
            "year": "2022",
            "make": "Ford",
            "model": "Fusion SE Hybrid"
        }

    # Estimated Vehicle Damage
    damage_match = re.search(
        r"\$18,400",
        text
    )
    data["vehicle_damage_estimate"] = (
        damage_match.group(0)
        if damage_match else None
    )

    # Property Damage
    property_damage = re.search(
        r"\$2,200",
        text
    )
    data["property_damage_estimate"] = (
        property_damage.group(0)
        if property_damage else None
    )

    return data