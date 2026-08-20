"""Author the internal source documents (Q2 input types).

The public web gives us product pages, an FAQ and policy pages. It cannot give
us the other half of what the assessment lists as input: a credit policy PDF, a
tabular rules matrix, an application form, an agent objection handbook, and
records containing PII. No lender publishes those.

So they are authored here, and they are honest about it: every file opens with
a SYNTHETIC banner, every record derived from them is tagged
`source_type: internal_*`, and the citation the bot speaks says so.

The messiness is deliberate and is the point of the exercise:

* `EMI` / `instalment` / `installment` / `monthly payment` used interchangeably
  -> exercises terminology standardisation
* three date formats -> exercises date normalisation
* a paragraph repeated near-verbatim in two documents -> exercises near-dup
* one deliberately truncated/garbled line -> exercises extraction-failure flags
* a lead export full of names, phones, emails, PAN and account numbers
  -> exercises PII detection and the retrieval block
"""
from __future__ import annotations

import csv
from pathlib import Path

import fitz  # PyMuPDF

from ...common.config import settings
from ...common.logging import log

BANNER = (
    "SYNTHETIC DOCUMENT - AUTHORED FOR A TECHNICAL ASSESSMENT. "
    "This is not a real lender's document and is not issued by, affiliated "
    "with, or endorsed by any company named in this repository. Figures are "
    "illustrative."
)

# ---------------------------------------------------------------------------
# 1. Credit policy (PDF)
# ---------------------------------------------------------------------------
CREDIT_POLICY_PAGES: list[tuple[str, str]] = [
    (
        "SME Credit Policy v2.1",
        BANNER
        + "\n\n"
        "Effective date: 01-04-2026. Supersedes v2.0 dated 2025/10/15.\n"
        "Owner: Credit Risk. Review cycle: annual.\n\n"
        "1. PURPOSE\n"
        "This policy defines the minimum qualification norms, documentation and "
        "pricing bands for unsecured and secured lending to micro, small and "
        "medium enterprises. It applies to all sourcing channels including "
        "branch, direct selling agents, branch partners and digital lead forms.\n\n"
        "2. PRODUCT SUITE\n"
        "2.1 Unsecured Business Loan: ticket size Rs. 3,00,000 to Rs. 50,00,000, "
        "tenure 12 to 36 months, no collateral, monthly EMI.\n"
        "2.2 Secured Business Loan against Property: ticket size Rs. 10,00,000 to "
        "Rs. 5,00,00,000, tenure up to 120 months, loan to value up to 65 percent "
        "of assessed market value.\n"
        "2.3 Machinery and Equipment Finance: up to 80 percent of invoice value, "
        "tenure 12 to 60 months, hypothecation of the financed asset.\n"
        "2.4 Supply Chain / Retailer Finance: revolving limit against verified "
        "anchor invoices, tenure 30 to 120 days per drawdown.\n",
    ),
    (
        "SME Credit Policy v2.1 - Eligibility",
        "3. ELIGIBILITY NORMS (UNSECURED BUSINESS LOAN)\n"
        "3.1 Business vintage: minimum 36 months of continuous operations under "
        "the same ownership, evidenced by GST registration or Udyam certificate.\n"
        "3.2 Annual turnover: minimum Rs. 40,00,000 for the last completed "
        "financial year, supported by GST returns.\n"
        "3.3 Applicant age: 24 years at application and not more than 65 years at "
        "loan maturity.\n"
        "3.4 Credit bureau score: 700 and above for the primary applicant. "
        "Between 675 and 699 the file may proceed only with a co-applicant "
        "and reduced ticket size.\n"
        "3.5 Banking conduct: last 12 months bank statements with average monthly "
        "credit of at least twice the proposed monthly payment. No more than "
        "two cheque returns in the last 6 months.\n"
        "3.6 Existing obligations: total fixed obligation to income ratio after "
        "the proposed instalment must not exceed 55 percent.\n"
        "3.7 Entity types accepted: proprietorship, partnership, LLP, private "
        "limited company. Not accepted: trusts, societies, HUF-only entities.\n\n"
        "4. NEGATIVE LIST\n"
        "Loans are not extended to: gambling and betting operations, crypto "
        "trading desks, unregistered money lending, arms and ammunition trade, "
        "tobacco manufacturing, and any activity on the applicable regulatory "
        "prohibited list.\n",
    ),
    (
        "SME Credit Policy v2.1 - Pricing and charges",
        "5. PRICING GRID (INDICATIVE)\n"
        "Risk grade A: 14.5% to 16.0% per annum. Processing fee 1.5% plus taxes.\n"
        "Risk grade B: 16.0% to 19.0% per annum. Processing fee 2.0% plus taxes.\n"
        "Risk grade C: 19.0% to 24.0% per annum. Processing fee 2.5% plus taxes.\n"
        "Interest is charged on a reducing balance basis. The monthly payment is "
        "collected by NACH mandate on the 5th of each month.\n\n"
        "6. OTHER CHARGES\n"
        "Bounce charge Rs. 500 per instance plus taxes. Penal charge 2% per month "
        "on the overdue instalment amount. Foreclosure charge 4% of principal "
        "outstanding, nil after 12 EMIs for individual borrowers on floating "
        "rate facilities. Duplicate statement Rs. 250.\n\n"
        "7. TURNAROUND TIME\n"
        "In-principle decision within 48 working hours of a complete file. "
        "Disbursement within 5 working days of sanction and documentation.\n\n"
        "8. DATA PROTECTION\n"
        "Customer identifiers collected during qualification calls - name, mobile "
        "number, PAN, GST number, bank details - are personal data. They must be "
        "stored only in the lending system of record, must never be repeated back "
        "in full on a recorded line, and are retained for the period required by "
        "applicable law and then purged.\n"
        "Recording disclosure is mandatory at the start of every outbound call, "
        "and where the caller is an automated assistant that fact must be "
        "disclosed before any qualification question is asked.\n",
    ),
    (
        "SME Credit Policy v2.1 - Collections and grievance",
        "9. COLLECTIONS CONDUCT\n"
        "Contact is permitted between 08:00 and 19:00 local time only. Agents may "
        "not use threatening language, may not contact references before the "
        "account is 30 days past due, and must offer the documented restructuring "
        "options before any escalation.\n"
        "Where a borrower states an inability to pay, the agent must offer the "
        "approved payment support path - part payment, date shift within the same "
        "month, or a callback from the resolution desk - and must not promise any "
        "waiver.\n\n"
        "10. GRIEVANCE REDRESSAL\n"
        "Level 1: branch or relationship manager, response within 7 days. "
        "Level 2: nodal officer, response within 15 days. Level 3: the regulator's "
        "ombudsman scheme. A customer who asks to speak to a human must be "
        "transferred or scheduled with a human officer on the same call.\n\n"
        "11. BRANCH PARTNER SUPPORT\n"
        "Operational, marketing and technology support is provided to branch "
        "partners, including lead management tooling, co-branded campaign material "
        "and product training. Partner payouts are processed monthly against "
        "disbursed and non-delinquent business.\n"
        "12. ANNEXURE - REVISION LOG\n"
        "v2.1 April 1, 2026 - revised bureau cut-off, added regional accent "
        "guidance for tele-calling. v2.0 2025/10/15 - initial digital release.\n"
        "v1.4 15-03-2025 - pricing grid revision [text truncated in source scan --",
    ),
]

# ---------------------------------------------------------------------------
# 2. Qualification rules matrix (CSV) - consumed directly by the voice agent
# ---------------------------------------------------------------------------
QUALIFICATION_RULES = [
    ["rule_id", "product", "slot", "operator", "value", "disposition", "reason_code", "customer_message"],
    ["QR001", "unsecured_business_loan", "entity_type", "in",
     "proprietorship|partnership|llp|private_limited", "pass", "ENTITY_OK",
     "Your business structure is eligible for our unsecured business loan."],
    ["QR002", "unsecured_business_loan", "entity_type", "in", "trust|society|huf",
     "reject", "ENTITY_NOT_SERVED",
     "We currently lend only to proprietorships, partnerships, LLPs and private limited companies."],
    ["QR003", "unsecured_business_loan", "business_vintage_months", "gte", "36", "pass",
     "VINTAGE_OK", "Your business vintage meets our requirement."],
    ["QR004", "unsecured_business_loan", "business_vintage_months", "between", "24|35", "refer",
     "VINTAGE_BORDERLINE",
     "Your vintage is slightly below our standard requirement, so a credit officer will review it."],
    ["QR005", "unsecured_business_loan", "business_vintage_months", "lt", "24", "reject",
     "VINTAGE_SHORT",
     "We need at least two years of business operations before we can consider an application."],
    ["QR006", "unsecured_business_loan", "annual_turnover_inr", "gte", "4000000", "pass",
     "TURNOVER_OK", "Your turnover is within our lending range."],
    ["QR007", "unsecured_business_loan", "annual_turnover_inr", "between", "2500000|3999999", "refer",
     "TURNOVER_BORDERLINE",
     "Your turnover is close to our threshold, so this needs a credit officer to review."],
    ["QR008", "unsecured_business_loan", "annual_turnover_inr", "lt", "2500000", "reject",
     "TURNOVER_LOW",
     "For this product we need annual turnover of at least twenty five lakh rupees."],
    ["QR009", "unsecured_business_loan", "loan_amount_inr", "between", "300000|5000000", "pass",
     "AMOUNT_OK", "That amount is within the range we offer without collateral."],
    ["QR010", "unsecured_business_loan", "loan_amount_inr", "gt", "5000000", "refer",
     "AMOUNT_HIGH",
     "Above fifty lakh we would look at a secured facility against property instead."],
    ["QR011", "unsecured_business_loan", "credit_score", "gte", "700", "pass", "BUREAU_OK",
     "Your credit profile looks suitable."],
    ["QR012", "unsecured_business_loan", "credit_score", "between", "675|699", "refer",
     "BUREAU_BORDERLINE",
     "With that score we would look at adding a co-applicant."],
    ["QR013", "unsecured_business_loan", "credit_score", "lt", "675", "reject", "BUREAU_LOW",
     "At present we are not able to proceed with a bureau score below 675."],
    ["QR014", "unsecured_business_loan", "gst_registered", "eq", "true", "pass", "GST_OK",
     "GST registration helps us verify turnover quickly."],
    ["QR015", "unsecured_business_loan", "gst_registered", "eq", "false", "refer", "GST_MISSING",
     "Without GST registration we would need Udyam registration and bank statements instead."],
    ["QR016", "unsecured_business_loan", "industry", "in",
     "gambling|crypto_trading|arms|tobacco_manufacturing|money_lending", "reject",
     "NEGATIVE_LIST", "That industry is on our restricted list, so we cannot proceed."],
    ["QR017", "unsecured_business_loan", "applicant_age", "between", "24|65", "pass", "AGE_OK",
     "Your age is within our policy range."],
    ["QR018", "unsecured_business_loan", "applicant_age", "outside", "24|65", "reject", "AGE_OUT",
     "Our policy requires the applicant to be between 24 and 65 years of age."],
    ["QR019", "loan_against_property", "loan_amount_inr", "between", "1000000|50000000", "pass",
     "LAP_AMOUNT_OK", "That is within the range for a loan against property."],
    ["QR020", "loan_against_property", "property_type", "in",
     "residential|commercial|industrial", "pass", "LAP_PROPERTY_OK",
     "That property type is acceptable as security."],
]

# ---------------------------------------------------------------------------
# 3. Objection handbook (markdown)
# ---------------------------------------------------------------------------
OBJECTION_HANDBOOK = BANNER + """

# Telecalling Objection Handbook - SME Lending

Version 1.3. Last revised April 1, 2026. Owner: Sales Enablement.

Every response below must stay inside policy. Agents may not quote a final
interest rate on the qualification call, may not promise sanction, and may not
waive any charge.

## OBJ-01 "Your interest rate is too high"
Acknowledge, then reframe on total cost and speed rather than arguing.
Say: the rate depends on the credit grade, turnover and banking conduct, and
the band starts at 14.5 percent per annum on a reducing balance basis. Point
out that there is no collateral requirement and that an in-principle decision
comes within 48 working hours. Never quote a single fixed rate on the call.

## OBJ-02 "I already have a loan running"
This is not a disqualifier. Existing obligations are allowed as long as the
total fixed obligation to income ratio after the new monthly payment stays
within 55 percent. Ask for the current EMI amount and the remaining tenure,
then continue qualification.

## OBJ-03 "Send me details, I will call back"
Agree, and pin the next step. Offer to send the product summary on WhatsApp or
email and propose a specific callback slot. Capture the preferred slot as a
callback request rather than closing the lead as not interested.

## OBJ-04 "I do not want to share my documents"
Explain what each document is used for: GST returns and bank statements are
used only to verify turnover and repayment capacity. Confirm that documents are
handled under the data protection policy and are not shared with third parties
for marketing. If the customer still refuses, offer a branch appointment.

## OBJ-05 "Another lender offered me a better rate"
Do not disparage the competitor. Ask what the offer includes: processing fee,
foreclosure charge, insurance loading and the disbursal timeline. Position on
total cost and turnaround, and offer to have a credit officer review the
competing sanction letter.

## OBJ-06 "Are there hidden charges?"
Answer directly and completely: processing fee, bounce charge, penal charge on
overdue instalments and foreclosure charge, all listed in the published
schedule of charges. Offer to send the schedule. Transparency here reduces
later disputes.

## OBJ-07 "I am not interested" (early in the call)
One recovery attempt only, then close politely. Ask whether the requirement is
timing or the product itself. If timing, capture a callback date. If the
customer repeats the refusal, thank them and end the call. Do not attempt a
third recovery.

## OBJ-08 "I want to talk to a person"
Stop qualification immediately. Confirm the request, capture the callback
number and preferred time, and hand off to the human desk. This request is
never to be argued with or deflected.

## Escalation triggers
Escalate to a human officer without attempting a rebuttal when the customer:
mentions a complaint or the ombudsman, uses distressed language, disputes an
existing account, alleges misselling, or asks the same question three times.
"""

# ---------------------------------------------------------------------------
# 4. Branch partner FAQ (markdown) - deliberately shares a near-duplicate
#    paragraph with the credit policy so dedup has something real to catch
# ---------------------------------------------------------------------------
PARTNER_FAQ = BANNER + """

# Branch Partner Programme - FAQ

Version 1.1. Effective 2026/04/01.

## What support do branch partners receive?
Operational, marketing and technology support is provided to branch partners,
including lead management tooling, co-branded campaign material and product
training. Partner payouts are processed monthly against disbursed and
non-delinquent business.

## Who can become a branch partner?
Existing loan consultants, chartered accountants and financial distributors
with at least two years of experience in SME lending and a registered business
entity.

## How are partner payouts calculated?
Payout is a percentage of the disbursed amount, banded by product and ticket
size, and is released in the monthly cycle following disbursement. Payout is
clawed back if the account turns delinquent within the first three instalments.

## What is the difference between a branch partner and a DSA?
A direct selling agent sources leads only. A branch partner additionally
handles first-level document collection and customer servicing, and operates
under a co-branded identity in the assigned territory.

## Do partners get access to the qualification rules?
Yes. Partners receive the current qualification matrix and the schedule of
charges, and must use only the approved communication templates.

## Can a partner quote an interest rate?
No. Partners and telecalling agents may share the published rate band only.
The final rate is decided by credit after underwriting.
"""

# ---------------------------------------------------------------------------
# 5. Application form (PDF)
# ---------------------------------------------------------------------------
APPLICATION_FORM_PAGES: list[tuple[str, str]] = [
    (
        "Business Loan Application Form",
        BANNER
        + "\n\nFORM BL-01  |  Version 4  |  Effective April 1, 2026\n\n"
        "SECTION A - APPLICANT DETAILS\n"
        "A1. Full name of applicant (as per PAN): ______________________\n"
        "A2. Date of birth (DD-MM-YYYY): ____________\n"
        "A3. Mobile number: ____________   A4. Email address: ____________\n"
        "A5. PAN: __________   A6. Aadhaar (last 4 digits only): ______\n"
        "A7. Residential address with PIN code: ______________________\n\n"
        "SECTION B - BUSINESS DETAILS\n"
        "B1. Registered business name: ______________________\n"
        "B2. Constitution (tick one): Proprietorship / Partnership / LLP / "
        "Private Limited\n"
        "B3. Business commencement date (DD-MM-YYYY): ____________\n"
        "B4. Industry / activity: ____________  B5. Udyam number: ____________\n"
        "B6. GST number: ____________  B7. Business address: ____________\n"
        "B8. Annual turnover last financial year (Rs.): ____________\n"
        "B9. Number of employees: ______\n",
    ),
    (
        "Business Loan Application Form - Facility and declarations",
        "SECTION C - FACILITY REQUESTED\n"
        "C1. Product: Unsecured Business Loan / Loan Against Property / "
        "Machinery Finance / Supply Chain Finance\n"
        "C2. Amount requested (Rs.): ____________\n"
        "C3. Preferred tenure in months: ______\n"
        "C4. Purpose of loan: working capital / machinery purchase / business "
        "expansion / inventory / debt consolidation\n"
        "C5. Existing loan obligations - lender, outstanding, monthly payment:\n"
        "    ______________________________________________\n\n"
        "SECTION D - DOCUMENT CHECKLIST\n"
        "D1. PAN and address proof of applicant and co-applicant\n"
        "D2. Business registration proof - Udyam or GST certificate\n"
        "D3. Last 12 months bank statements of the primary business account\n"
        "D4. Last 2 years income tax returns with computation\n"
        "D5. Last 4 quarters GST returns\n"
        "D6. Property documents, for a loan against property only\n\n"
        "SECTION E - DECLARATIONS AND CONSENT\n"
        "E1. I confirm the information given is true and complete.\n"
        "E2. I authorise a credit bureau enquiry.\n"
        "E3. I consent to being contacted on the number above, including by an "
        "automated assistant, and understand calls may be recorded.\n"
        "E4. I have read the schedule of charges and the fair practices code.\n\n"
        "Signature: ____________   Place: ____________   Date: ____________\n",
    ),
]

# ---------------------------------------------------------------------------
# 6. CRM lead export (CSV) - the PII test set
# ---------------------------------------------------------------------------
LEAD_EXPORT = [
    ["lead_id", "captured_on", "full_name", "mobile", "email", "pan", "aadhaar",
     "business_name", "city", "turnover_inr", "amount_requested_inr", "agent_note"],
    ["LD-10241", "01-04-2026", "Ramesh Iyer", "+91 98204 33127", "ramesh.iyer@sharadatextiles.in",
     "AFZPK7190K", "4321 8765 2190", "Sharada Textiles", "Surat", "7200000", "1500000",
     "Wants working capital before festive season. Existing EMI 42000 with HDFC."],
    ["LD-10242", "April 2, 2026", "Fatima Sheikh", "9845012278", "fatima.sheikh92@gmail.com",
     "BKLPS4432M", "", "Sheikh Auto Spares", "Hyderabad", "3100000", "800000",
     "Turnover borderline. Asked to call back after 6 pm."],
    ["LD-10243", "2026/04/02", "Harpreet Singh Bedi", "+91-98110-77451", "hs.bedi@bediagro.co.in",
     "CQWPB9981L", "7788 1122 3344", "Bedi Agro Equipments", "Ludhiana", "18500000", "4500000",
     "Machinery finance. Account 50100234456789 with ICICI for NACH."],
    ["LD-10244", "03-04-2026", "Meena Krishnan", "8892340012", "meena@krishnanpharma.com",
     "DLMPK2210J", "", "Krishnan Pharma Distributors", "Coimbatore", "9600000", "2000000",
     "Asked about foreclosure charges twice. Rate sensitive."],
    ["LD-10245", "03-04-2026", "Abdul Rahman", "+91 99400 11298", "abdul.rahman@rahmantraders.in",
     "EPQPR5567H", "9911 2233 4455", "Rahman Traders", "Chennai", "2100000", "600000",
     "Below turnover cut-off. Requested human callback."],
    ["LD-10246", "April 4, 2026", "Nikhil Deshpande", "9822114567", "nikhil.d@deshpandeprint.in",
     "FRSPD8823G", "", "Deshpande Printers", "Pune", "5400000", "1200000",
     "Competitor offer from Bajaj at 15.2 percent. Wants comparison."],
    ["LD-10247", "04-04-2026", "Lakshmi Narayanan", "+91 90035 22190", "lakshmi@lncatering.co.in",
     "GTUPN1145F", "3344 5566 7788", "LN Catering Services", "Bengaluru", "4800000", "900000",
     "Cheque bounce twice in last 6 months. Needs credit review."],
    ["LD-10248", "05-04-2026", "Sanjay Gupta", "9910044321", "sanjay.gupta@guptaelectricals.in",
     "HVWPG3378D", "", "Gupta Electricals", "Delhi", "11200000", "3000000",
     "GST registered. Clean banking. Priority follow-up."],
    ["LD-10249", "2026/04/05", "Priya Menon", "+91 98470 66512", "priya.menon@menonexports.com",
     "IXYPM7734C", "5566 7788 9900", "Menon Exports", "Kochi", "23000000", "6000000",
     "Above unsecured cap. Route to LAP desk."],
    ["LD-10250", "06-04-2026", "Vikram Chauhan", "8800123499", "vikram@chauhanlogistics.in",
     "JZAPC2256B", "", "Chauhan Logistics", "Jaipur", "6700000", "1800000",
     "Trust structure - not eligible per policy. Informed politely."],
    ["LD-10251", "April 6, 2026", "Ananya Bose", "+91 98301 44872", "ananya.bose@boseinteriors.in",
     "KBCPB9912A", "1122 3344 5566", "Bose Interiors", "Kolkata", "3900000", "1000000",
     "Interested but wants to speak to a person before sharing documents."],
    ["LD-10252", "07-04-2026", "Imran Qureshi", "9769012234", "imran.q@qureshileather.in",
     "LDEPQ4478Z", "", "Qureshi Leather Works", "Kanpur", "8300000", "2200000",
     "Asked whether the caller was a robot. Disclosure given."],
]


def _write_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    doc = fitz.open()
    for title, body in pages:
        page = doc.new_page()
        page.insert_text((60, 60), title, fontsize=14, fontname="hebo")
        rect = fitz.Rect(60, 85, 545, 780)
        page.insert_textbox(rect, body, fontsize=9.2, fontname="helv", lineheight=1.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def seed() -> dict[str, Path]:
    base = settings.raw_dir / "internal"
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    p = base / "credit_policy_v2.pdf"
    _write_pdf(p, CREDIT_POLICY_PAGES)
    written["credit_policy"] = p

    p = base / "loan_application_form.pdf"
    _write_pdf(p, APPLICATION_FORM_PAGES)
    written["application_form"] = p

    p = base / "qualification_rules.csv"
    _write_csv(p, QUALIFICATION_RULES)
    written["qualification_rules"] = p

    p = base / "crm_lead_export.csv"
    _write_csv(p, LEAD_EXPORT)
    written["lead_export"] = p

    p = base / "agent_objection_handbook.md"
    p.write_text(OBJECTION_HANDBOOK, encoding="utf-8")
    written["objection_handbook"] = p

    p = base / "branch_partner_faq.md"
    p.write_text(PARTNER_FAQ, encoding="utf-8")
    written["partner_faq"] = p

    for name, path in written.items():
        log("seed.written", doc=name, path=str(path), bytes=path.stat().st_size)
    return written


if __name__ == "__main__":
    seed()
