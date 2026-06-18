import re
import json
from typing import Optional


def extract_acord_fields(raw_text: str) -> dict:
    """
    Extract all labelled fields from ACORD form OCR text.

    Args:
        raw_text: Raw string extracted from scanned ACORD PDF.

    Returns:
        dict: Structured JSON-serialisable dictionary of all fields.
    """

    def find(pattern: str, text: str, group: int = 1) -> Optional[str]:
        """Return first regex match or None, stripped."""
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(group).strip() if m else None

    # ------------------------------------------------------------------ #
    #  CLIENT / INSURED DETAILS
    # ------------------------------------------------------------------ #
    client = {
        "insured_name":   find(r"INSURED:\s+([A-Z\s]+CLIENT)", raw_text),
        "item_id":        find(r"ITEM ID[:\s]+(\d+)", raw_text),
        "client_code":    find(r"Client Code\s+\d?\s+(\w+)", raw_text),
        "bill_to":        find(r"Bill To\s+['\w]+\s+Q?\s+'?([^\n]+)", raw_text),
        "division":       find(r"Division\s+\d+\s+Q\s+(.+?)(?:\n|$)", raw_text),
        "last_entry":     find(r"Last Entry\s+([\d/]+)", raw_text),
        "last_user":      find(r"Last User\s+'?(\w+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  ADDRESS
    # ------------------------------------------------------------------ #
    address = {
        "address_line_1": find(r"Address #1[^[]*\[([^\]\n]{1,60})\]", raw_text)
                          or find(r"Address #1\s+\d+\s+\[?(\d+\s*\w[\w\s]{0,40}?)(?:\n|Address)", raw_text),
        "zip_code":       find(r"Zip Code\s+\d+\s+(\d{4,10})", raw_text),
        "state":          find(r"\bState\s+([A-Z]{2})\b", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  PHONE / CONTACT
    # ------------------------------------------------------------------ #
    phones = []
    for m in re.finditer(
        r"Phone #(\d+)\s+\d+\s+([\d\s\-]+)\s+Extension\s+\d+\s+([\d]*)\s+Phone Type\s+\d+\s+Q\s+(\w+)",
        raw_text, re.IGNORECASE
    ):
        phones.append({
            "phone_number": m.group(2).strip(),
            "extension":    m.group(3).strip() or None,
            "phone_type":   m.group(4).strip(),
        })

    contact = {
        "phones":           phones,
        "fax_number":       find(r"Fax Number\s+([\d]+)", raw_text),
        "contact_method":   find(r"Contact Method\s+(\w+)", raw_text),
        "website":          find(r"Website\s+(www\.\S+)", raw_text),
        "fein":             find(r"FEIN\s+\d*\s+([\d]+)", raw_text),
        "legal_entity":     find(r"Legal Entity\s+\w+\s+\|?Q\s+(\w+)", raw_text),
        "date_biz_started": find(r"Date Business Started\s+([\d/]+)", raw_text),
        "years_in_business":find(r"Years in Business\s+(\d+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  PRIMARY CONTACT
    # ------------------------------------------------------------------ #
    primary_contact = {
        "first_name":    find(r"FirstName\s+\d+\s+(\S+)", raw_text),
        "middle_name":   find(r"Middle Name\s+(\S+)", raw_text),
        "last_name":     find(r"Last Name\s+(\w+)", raw_text),
        "salutation":    find(r"Salutation\s+'?(\w+\.?)", raw_text),
        "profession":    find(r"Contact Profession\s+(\w+)", raw_text),
        "primary_phone": find(r"PrimaryPhone\s+\d+\s+([\d\-]+)", raw_text),
        "phone_ext":     find(r"PrimaryPhone[^\n]+Ext:\s*(\d+)", raw_text),
        "primary_email": find(r"PrimaryEmail\s+\d+\s+(\S+@\S+)", raw_text),
        "primary_address": find(r"Primary Address\s+\d+\s+([^\n]+)", raw_text),
        "contact_type":  find(r"Contact Type\s+\w+\s+\|?Q\s+(\w+)", raw_text),
        "contact_codes": find(r"Contact Codes\s+([\w\-\(\)]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  POLICY DETAILS
    # ------------------------------------------------------------------ #
    policy = {
        "policy_number":          find(r"Policy Number\s+\d+\s+=?\s*(\w+)", raw_text),
        "effective_date":         find(r"Effective Date\s+\(?([\d/]+)", raw_text),
        "expiration_date":        find(r"Expiration Date\s+\(?([\d/]+)", raw_text),
        "original_effective_date":find(r"Original Effective Date\s+\(?([\d/]+)", raw_text),
        "description":            find(r"Description\s+([^\n]+?)(?:Mapping|$)", raw_text),
        "division_agency":        find(r"Division\s+\d+\s+QQ\s+([^\n]+)", raw_text),
        "commission_pct":         find(r"Commission\s*%\s*([\d.]+)", raw_text),
        "sic_code":               find(r"SIC Code\s+(\d+)", raw_text),
        "sic_description":        find(r"SIC Code\s+\d+\s+Q\s+([^\n]+)", raw_text),
        "policy_type":            find(r"Policy Type\s+\w+\s+Q\s+(\w+)", raw_text),
        "coverage":               find(r"Coverage\s+\d+\s+\[?\w+\]?\s+\|?&?\s+([^\n]+)", raw_text),
        "bill_method":            find(r"Bill Method\s+\[([^\]]+)\]", raw_text),
        "term":                   find(r"\bTerm\s+\[([A-Za-z\-]+)", raw_text),
        "audit_term":             find(r"Audit Term\s+(A-\w+|Annual|Monthly|Quarterly)", raw_text),
        "status":                 find(r"\bStatus\s+(\w+)", raw_text),
        "payment_plan":           find(r"Payment Plan\s+\w+\s+\|?Q\s+(\w+)", raw_text),
        "carrier":                find(r"(The Hartford Ins\. Group)", raw_text),
        "producer":               find(r"Producers?\s+\d*\s+([^\n]+)", raw_text),
        "servicer":               find(r"Servicer\s+\w+\s+\|?Q\s+([^\n]+)", raw_text),
        "source_date":            find(r"Source Date\s+\(?([\d/]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  CLAIMS / LOSS
    # ------------------------------------------------------------------ #
    loss = {
        "loss_id":          find(r"Loss I\.?D\.?\s+(\w+)", raw_text),
        "loss_type":        find(r"LossType\s+\d+[^@\n]+?(\w[\w\s]+Loss)", raw_text),
        "loss_amount":      find(r"Loss Amount\s+([\d,]+)", raw_text),
        "loss_date":        find(r"LossDate\s+\d+\s+\(?([\d/]+)", raw_text),
        "loss_time":        find(r"LossTime\s+\d+\s+([\d:apm]+)", raw_text),
        "reported_date":    find(r"Reported Date\s+\(?([\d/]+)", raw_text),
        "loss_location":    find(r"Location\?\s+\d+\s+\d+\s+Q\s+([^;]+)", raw_text),
        "loss_city":        find(r"City\s+([A-Za-z\s]+)\s+State", raw_text),
        "description_of_loss": find(r"Description of Loss Field\s+([^\n]+)", raw_text),
        "cat_loss":         find(r"Cat Loss\s+(\w+)\b(?!\d{2})", raw_text) or find(r"CatLoss:([YN])", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  ADJUSTER INFO
    # ------------------------------------------------------------------ #
    adjuster = {
        "name":      find(r"Adjuster Name\s+\d+\s+\|?\s*([^\n]+)", raw_text),
        "phone":     find(r"Adjuster Phone\s+\d+\s+([\d\s\-]+)", raw_text),
        "phone_ext": find(r"Adjuster Phone Ext\s+\d+\s+(\d+)", raw_text),
        "email":     find(r"Adjuster Email\s+\d+\s+(\S+@\S+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  CLAIMANT
    # ------------------------------------------------------------------ #
    claimant = {
        "name":           find(r"CLAIMANT NAME\s*\n[^\n]*\n\s*\d+\s+[A-Z]+\s+\*+\s+\w+\s+([^\n]+)", raw_text),
        "address":        find(r"(\d+\s+\w+\s+St\.?,\s*\w+,?\s*\w+\s*\d+).*?36", raw_text),
        "home_phone":     find(r"Home Phone\s*#\s*\d+\s+([\d\-]+)", raw_text),
        "business_phone": find(r"Business Phone\s*#\s*\d+\s+([\d\-]+)", raw_text),
        "ssn_fein":       find(r"SSNE?\s+(\w+)", raw_text),
        "status":         find(r"Status\s+°?\s+Q\s+([\w\s]+?)(?:\s+Home Phone|\s+Type|\n|$)", raw_text),
        "type_of_loss":   find(r"Type of Loss\s+\w\s+Q\s+([\w\s]+?)(?:\s+Business Phone|\s+Injured|\n|$)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  ACORD FORM META
    # ------------------------------------------------------------------ #
    form_meta = {
        "agency":             find(r"(The ABC General Agency,\s*Inc\.?)", raw_text),
        "agency_customer_id": find(r"AGENCY CUSTOMER ID[:\s]+(\w+)", raw_text),
        "form_number":        find(r"FORM NUMBER:\s+(\d+)", raw_text),
        "form_title":         find(r"FORM TITLE:\s+([^\n]+)", raw_text),
        "producer_address":   find(r"P\.O\. Box\s+\d+\s+\n?\s*([^\n]+)", raw_text),
        "reported_by":        find(r"REPORTED BY\s+([^\n]+)", raw_text),
        "reported_to":        find(r"REPORTED TO\s+([^\n]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  ASSEMBLE OUTPUT
    # ------------------------------------------------------------------ #
    result = {
        "client":          client,
        "address":         address,
        "contact":         contact,
        "primary_contact": primary_contact,
        "policy":          policy,
        "loss":            loss,
        "adjuster":        adjuster,
        "claimant":        claimant,
        "form_meta":       form_meta,
    }

    # Remove None values recursively for a cleaner output
    def drop_none(obj):
        if isinstance(obj, dict):
            return {k: drop_none(v) for k, v in obj.items() if v not in (None, "", [])}
        if isinstance(obj, list):
            return [drop_none(i) for i in obj if i not in (None, "", {})]
        return obj

    return drop_none(result)