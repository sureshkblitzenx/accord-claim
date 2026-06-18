import re
from typing import Optional


def extract_acord_fields(raw_text: str) -> dict:
    """
    Extract ACORD GL form fields and return them in the GL payload
    transformation format.

    Args:
        raw_text: Raw string extracted from scanned ACORD PDF.

    Returns:
        dict: Flat/nested JSON matching the GL payload schema.
    """

    def find(pattern: str, text: str, group: int = 1) -> Optional[str]:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(group).strip() if m else None

    # ------------------------------------------------------------------ #
    #  POLICY / HEADER
    # ------------------------------------------------------------------ #
    policy = {
        "AccountNumber":        find(r"AGENCY CUSTOMER ID[:\s]+(\w+)", raw_text),
        "PolicyNumber":         find(r"Policy Number\s+\d+\s+=?\s*(\w+)", raw_text),
        "EffectiveDate":        find(r"Effective Date\s+\(?([\d/]+)", raw_text),
        "ExpirationDate":       find(r"Expiration Date\s+\(?([\d/]+)", raw_text),
        "OriginalEffectiveDate":find(r"Original Effective Date\s+\(?([\d/]+)", raw_text),
        "PolicyType":           find(r"Policy Type\s+\w+\s+Q\s+(\w+)", raw_text),
        "LineOfBusiness":       find(r"Coverage\s+\d+\s+\[?\w+\]?\s+\|?&?\s+([^\n]+)", raw_text),
        "BillMethod":           find(r"Bill Method\s+\[([^\]]+)\]", raw_text),
        "PaymentPlan":          find(r"Payment Plan\s+\w+\s+\|?Q\s+(\w+)", raw_text),
        "Term":                 find(r"\bTerm\s+\[([A-Za-z\-]+)", raw_text),
        "AuditTerm":            find(r"Audit Term\s+(A-\w+|Annual|Monthly|Quarterly)", raw_text),
        "Status":               find(r"\bStatus\s+(\w+)", raw_text),
        "CommissionPct":        find(r"Commission\s*%\s*([\d.]+)", raw_text),
        "SICCode":              find(r"SIC Code\s+(\d+)", raw_text),
        "SICDescription":       find(r"SIC Code\s+\d+\s+Q\s+([^\n]+)", raw_text),
        "Description":          find(r"Description\s+([^\n]+?)(?:Mapping|$)", raw_text),
        "Producer":             find(r"Producers?\s+\d*\s+([^\n]+)", raw_text),
        "Servicer":             find(r"Servicer\s+\w+\s+\|?Q\s+([^\n]+)", raw_text),
        "SourceDate":           find(r"Source Date\s+\(?([\d/]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  INSURED / NAMED INSURED
    # ------------------------------------------------------------------ #
    named_insured = {
        "InsuredName":    find(r"INSURED:\s+([A-Z\s]+CLIENT)", raw_text),
        "ItemID":         find(r"ITEM ID[:\s]+(\d+)", raw_text),
        "ClientCode":     find(r"Client Code\s+\d?\s+(\w+)", raw_text),
        "BillTo":         find(r"Bill To\s+['\w]+\s+Q?\s+'?([^\n]+)", raw_text),
        "Division":       find(r"Division\s+\d+\s+Q\s+(.+?)(?:\n|$)", raw_text),
        "LastEntry":      find(r"Last Entry\s+([\d/]+)", raw_text),
        "LastUser":       find(r"Last User\s+'?(\w+)", raw_text),
        "FEIN":           find(r"FEIN\s+\d*\s+([\d]+)", raw_text),
        "LegalEntity":    find(r"Legal Entity\s+\w+\s+\|?Q\s+(\w+)", raw_text),
        "DateBizStarted": find(r"Date Business Started\s+([\d/]+)", raw_text),
        "YearsInBusiness":find(r"Years in Business\s+(\d+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  MAILING ADDRESS
    # ------------------------------------------------------------------ #
    mailing_address = {
        "AddressLine1": (
            find(r"Address #1[^[]*\[([^\]\n]{1,60})\]", raw_text)
            or find(r"Address #1\s+\d+\s+\[?(\d+\s*\w[\w\s]{0,40}?)(?:\n|Address)", raw_text)
        ),
        "ZipCode": find(r"Zip Code\s+\d+\s+(\d{4,10})", raw_text),
        "State":   find(r"\bState\s+([A-Z]{2})\b", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  PHONES
    # ------------------------------------------------------------------ #
    phones = []
    for m in re.finditer(
        r"Phone #(\d+)\s+\d+\s+([\d\s\-]+)\s+Extension\s+\d+\s+([\d]*)\s+Phone Type\s+\d+\s+Q\s+(\w+)",
        raw_text, re.IGNORECASE
    ):
        phones.append({
            "PhoneNumber": m.group(2).strip(),
            "Extension":   m.group(3).strip() or None,
            "PhoneType":   m.group(4).strip(),
        })

    # ------------------------------------------------------------------ #
    #  CONTACT
    # ------------------------------------------------------------------ #
    contact = {
        "Phones":          phones,
        "FaxNumber":       find(r"Fax Number\s+([\d]+)", raw_text),
        "ContactMethod":   find(r"Contact Method\s+(\w+)", raw_text),
        "Website":         find(r"Website\s+(www\.\S+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  PRIMARY CONTACT
    # ------------------------------------------------------------------ #
    primary_contact = {
        "FirstName":      find(r"FirstName\s+\d+\s+(\S+)", raw_text),
        "MiddleName":     find(r"Middle Name\s+(\S+)", raw_text),
        "LastName":       find(r"Last Name\s+(\w+)", raw_text),
        "Salutation":     find(r"Salutation\s+'?(\w+\.?)", raw_text),
        "Profession":     find(r"Contact Profession\s+(\w+)", raw_text),
        "PrimaryPhone":   find(r"PrimaryPhone\s+\d+\s+([\d\-]+)", raw_text),
        "PhoneExt":       find(r"PrimaryPhone[^\n]+Ext:\s*(\d+)", raw_text),
        "PrimaryEmail":   find(r"PrimaryEmail\s+\d+\s+(\S+@\S+)", raw_text),
        "PrimaryAddress": find(r"Primary Address\s+\d+\s+([^\n]+)", raw_text),
        "ContactType":    find(r"Contact Type\s+\w+\s+\|?Q\s+(\w+)", raw_text),
        "ContactCodes":   find(r"Contact Codes\s+([\w\-\(\)]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  CARRIER
    # ------------------------------------------------------------------ #
    carrier = {
        "CarrierName": find(r"(The Hartford Ins\. Group)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  GENERAL LIABILITY COVERAGES
    # ------------------------------------------------------------------ #

    # Each occurrence / aggregate limit pulled by label
    gl_coverages = {
        "OccurrenceLimit":              find(r"Each Occurrence\s+\$?([\d,]+)", raw_text),
        "GeneralAggregateLimit":        find(r"General Aggregate\s+\$?([\d,]+)", raw_text),
        "ProductsCompOpsAggregate":     find(r"Products\s*[-–]\s*Comp[^\n]*\$?([\d,]+)", raw_text),
        "PersonalAdvInjuryLimit":       find(r"Personal\s*&?\s*Adv\s*Injury\s+\$?([\d,]+)", raw_text),
        "FireDamageLegalLiability":     find(r"Fire Damage\s+\$?([\d,]+)", raw_text),
        "MedicalExpenseLimit":          find(r"Med\s*Exp[^\n]*\$?([\d,]+)", raw_text),
        "Deductible":                   find(r"Deductible\s+\$?([\d,]+)", raw_text),
        "PremiumBasis":                 find(r"Premium Basis\s+([^\n]+)", raw_text),
        "ClassCode":                    find(r"Class\s*Code\s+(\d+)", raw_text),
        "ClassDescription":             find(r"Classification\s+([^\n]+)", raw_text),
        "Exposure":                     find(r"\bExposure\s+([\d,]+)", raw_text),
        "Rate":                         find(r"\bRate\s+([\d.]+)", raw_text),
        "AdvancePremium":               find(r"Advance\s+Premium\s+\$?([\d,]+)", raw_text),
        "AuditedPremium":               find(r"Audited\s+Premium\s+\$?([\d,]+)", raw_text),
        "WrittenPremium":               find(r"Written\s+Premium\s+\$?([\d,]+)", raw_text),
        "PolicyPremium":                find(r"Policy\s+Premium\s+\$?([\d,]+)", raw_text),
        "MinimumPremium":               find(r"Minimum\s+Premium\s+\$?([\d,]+)", raw_text),
        "DepositPremium":               find(r"Deposit\s+Premium\s+\$?([\d,]+)", raw_text),
        "AdditionalInsuredEndorsement": find(r"Additional Insured\s+([^\n]+)", raw_text),
        "WaiverOfSubrogation":          find(r"Waiver of Subrogation\s+([^\n]+)", raw_text),
        "UmbrellaRequired":             find(r"Umbrella\s+Required\s+(\w+)", raw_text),
        "UmbrellaCarrier":              find(r"Umbrella\s+Carrier\s+([^\n]+)", raw_text),
        "UmbrellaLimit":                find(r"Umbrella\s+Limit\s+\$?([\d,]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  LOCATIONS
    # ------------------------------------------------------------------ #
    locations = []
    for m in re.finditer(
        r"Location\s+(\d+)[^\n]*\n\s*([^\n]+)\n\s*([^\n]+)",
        raw_text, re.IGNORECASE
    ):
        locations.append({
            "LocationNumber": m.group(1).strip(),
            "AddressLine1":   m.group(2).strip(),
            "CityStateZip":   m.group(3).strip(),
        })

    # ------------------------------------------------------------------ #
    #  LOSS / CLAIMS
    # ------------------------------------------------------------------ #
    loss = {
        "LossID":            find(r"Loss I\.?D\.?\s+(\w+)", raw_text),
        "LossType":          find(r"LossType\s+\d+[^@\n]+?(\w[\w\s]+Loss)", raw_text),
        "LossAmount":        find(r"Loss Amount\s+([\d,]+)", raw_text),
        "LossDate":          find(r"LossDate\s+\d+\s+\(?([\d/]+)", raw_text),
        "LossTime":          find(r"LossTime\s+\d+\s+([\d:apm]+)", raw_text),
        "ReportedDate":      find(r"Reported Date\s+\(?([\d/]+)", raw_text),
        "LossLocation":      find(r"Location\?\s+\d+\s+\d+\s+Q\s+([^;]+)", raw_text),
        "LossCity":          find(r"City\s+([A-Za-z\s]+)\s+State", raw_text),
        "DescriptionOfLoss": find(r"Description of Loss Field\s+([^\n]+)", raw_text),
        "CatLoss":           find(r"Cat Loss\s+(\w+)\b(?!\d{2})", raw_text) or find(r"CatLoss:([YN])", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  ADJUSTER
    # ------------------------------------------------------------------ #
    adjuster = {
        "AdjusterName":     find(r"Adjuster Name\s+\d+\s+\|?\s*([^\n]+)", raw_text),
        "AdjusterPhone":    find(r"Adjuster Phone\s+\d+\s+([\d\s\-]+)", raw_text),
        "AdjusterPhoneExt": find(r"Adjuster Phone Ext\s+\d+\s+(\d+)", raw_text),
        "AdjusterEmail":    find(r"Adjuster Email\s+\d+\s+(\S+@\S+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  CLAIMANT
    # ------------------------------------------------------------------ #
    claimant = {
        "ClaimantName":     find(r"CLAIMANT NAME\s*\n[^\n]*\n\s*\d+\s+[A-Z]+\s+\*+\s+\w+\s+([^\n]+)", raw_text),
        "ClaimantAddress":  find(r"(\d+\s+\w+\s+St\.?,\s*\w+,?\s*\w+\s*\d+).*?36", raw_text),
        "HomePhone":        find(r"Home Phone\s*#\s*\d+\s+([\d\-]+)", raw_text),
        "BusinessPhone":    find(r"Business Phone\s*#\s*\d+\s+([\d\-]+)", raw_text),
        "SSNFEIN":          find(r"SSNE?\s+(\w+)", raw_text),
        "ClaimantStatus":   find(r"Status\s+°?\s+Q\s+([\w\s]+?)(?:\s+Home Phone|\s+Type|\n|$)", raw_text),
        "TypeOfLoss":       find(r"Type of Loss\s+\w\s+Q\s+([\w\s]+?)(?:\s+Business Phone|\s+Injured|\n|$)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  FORM META
    # ------------------------------------------------------------------ #
    form_meta = {
        "Agency":           find(r"(The ABC General Agency,\s*Inc\.?)", raw_text),
        "AgencyCustomerID": find(r"AGENCY CUSTOMER ID[:\s]+(\w+)", raw_text),
        "FormNumber":       find(r"FORM NUMBER:\s+(\d+)", raw_text),
        "FormTitle":        find(r"FORM TITLE:\s+([^\n]+)", raw_text),
        "ProducerAddress":  find(r"P\.O\. Box\s+\d+\s+\n?\s*([^\n]+)", raw_text),
        "ReportedBy":       find(r"REPORTED BY\s+([^\n]+)", raw_text),
        "ReportedTo":       find(r"REPORTED TO\s+([^\n]+)", raw_text),
    }

    # ------------------------------------------------------------------ #
    #  ASSEMBLE — matches GL payload transformation schema
    # ------------------------------------------------------------------ #
    result = {
        "Policy":          policy,
        "NamedInsured":    named_insured,
        "MailingAddress":  mailing_address,
        "Contact":         contact,
        "PrimaryContact":  primary_contact,
        "Carrier":         carrier,
        "GLCoverages":     gl_coverages,
        "Locations":       locations,
        "Loss":            loss,
        "Adjuster":        adjuster,
        "Claimant":        claimant,
        "FormMeta":        form_meta,
    }

    def drop_none(obj):
        if isinstance(obj, dict):
            return {k: drop_none(v) for k, v in obj.items() if v not in (None, "", [])}
        if isinstance(obj, list):
            return [drop_none(i) for i in obj if i not in (None, "", {})]
        return obj

    return drop_none(result)