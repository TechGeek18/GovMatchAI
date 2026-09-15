import json
import re
from datetime import datetime, timedelta

import requests
import streamlit as st


SAM_SEARCH_URL = "https://api.sam.gov/opportunities/v2/search"
GEMINI_MODEL = "gemini-1.5-flash"

st.set_page_config(page_title="GovMatch AI", page_icon="🏛️", layout="wide")
st.title("🏛️ GovMatch AI - Internal Pipeline")
st.write(
    "Search SAM.gov by the NAICS or Product Service Codes you choose, then review fit, projected value, "
    "place of performance, and submission details."
)

st.sidebar.header("⚙️ Search configuration")
sam_key = st.sidebar.text_input("SAM.gov API key", type="password")
llm_key = st.sidebar.text_input("Gemini API key", type="password")
naics_input = st.sidebar.text_area(
    "NAICS codes (comma, space, or one per line)",
    "541511, 541512, 541519",
    help="The app runs a separate SAM.gov search for every valid code you enter.",
)
psc_input = st.sidebar.text_area(
    "Product Service Codes (comma, space, or one per line)",
    "",
    help="Enter 1-4 letter/number PSC values, such as D302 or R425. Searches are combined with the NAICS results.",
)
lookback_days = st.sidebar.number_input("Days of postings to scan", 1, 365, 30)
max_per_code = st.sidebar.number_input("Maximum results per search code", 1, 1000, 50)
st.sidebar.markdown("---")
company_profile = st.sidebar.text_area(
    "Company profile (for AI matching)",
    "We are an IT consulting firm specializing in cloud architecture (AWS/Azure), "
    "custom Python/React development, and Zero Trust cybersecurity.",
)


def parse_naics_codes(raw_value):
    """Return unique, validated 2-6 digit NAICS codes while preserving order."""
    tokens = [item for item in re.split(r"[\s,;]+", raw_value.strip()) if item]
    valid, invalid = [], []
    for token in tokens:
        if re.fullmatch(r"\d{2,6}", token):
            if token not in valid:
                valid.append(token)
        else:
            invalid.append(token)
    return valid, invalid


def parse_psc_codes(raw_value):
    """Return unique, validated PSC/classification codes in uppercase."""
    tokens = [item.upper() for item in re.split(r"[\s,;]+", raw_value.strip()) if item]
    valid, invalid = [], []
    for token in tokens:
        if re.fullmatch(r"[A-Z0-9]{1,4}", token):
            if token not in valid:
                valid.append(token)
        else:
            invalid.append(token)
    return valid, invalid


def fetch_contracts(api_key, naics_codes, psc_codes, days, per_code):
    """Search and paginate independently for each user-entered NAICS or PSC code."""
    posted_to = datetime.now()
    posted_from = posted_to - timedelta(days=int(days))
    results = {}
    errors = []

    searches = [("NAICS", "ncode", code) for code in naics_codes]
    searches += [("PSC", "ccode", code) for code in psc_codes]

    for search_type, parameter_name, searched_code in searches:
        offset = 0
        remaining = int(per_code)
        while remaining > 0:
            limit = min(remaining, 1000)
            params = {
                "api_key": api_key,
                parameter_name: searched_code,
                "postedFrom": posted_from.strftime("%m/%d/%Y"),
                "postedTo": posted_to.strftime("%m/%d/%Y"),
                "limit": limit,
                "offset": offset,
            }
            try:
                response = requests.get(SAM_SEARCH_URL, params=params, timeout=30)
                response.raise_for_status()
                page = response.json().get("opportunitiesData", [])
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{search_type} {searched_code}: {exc}")
                break

            for opportunity in page:
                unique_id = opportunity.get("noticeId") or opportunity.get("solicitationNumber")
                unique_id = unique_id or f"{opportunity.get('title')}:{opportunity.get('postedDate')}"
                if unique_id not in results:
                    opportunity["_matched_naics"] = []
                    opportunity["_matched_psc"] = []
                    results[unique_id] = opportunity
                match_key = "_matched_naics" if search_type == "NAICS" else "_matched_psc"
                if searched_code not in results[unique_id][match_key]:
                    results[unique_id][match_key].append(searched_code)

            received = len(page)
            remaining -= received
            if received < limit:
                break
            offset += received

    return list(results.values()), errors


def nested_name(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("code") or ""
    return value or ""


def format_address(address):
    if not isinstance(address, dict):
        return "Not published"
    pieces = [
        address.get("streetAddress") or address.get("streetAddress1"),
        address.get("streetAddress2"),
        nested_name(address.get("city")),
        nested_name(address.get("state")),
        address.get("zip") or address.get("zipcode"),
        nested_name(address.get("country")) or address.get("countryCode"),
    ]
    return ", ".join(str(piece) for piece in pieces if piece) or "Not published"


def currency(value):
    try:
        return f"${float(str(value).replace(',', '')):,.0f}"
    except (TypeError, ValueError):
        return "Not published"


def clean_json_response(raw_text):
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw_text)
    return re.sub(r"\s*```\s*$", "", cleaned).strip()


def analyze_with_gemini(opportunity, gemini_key, profile):
    """Score fit and extract/estimate details that are not structured by SAM.gov."""
    prompt_record = {
        "title": opportunity.get("title"),
        "description": opportunity.get("description"),
        "notice_type": opportunity.get("type"),
        "naics": opportunity.get("naicsCode"),
        "classification_code": opportunity.get("classificationCode"),
        "response_deadline": opportunity.get("responseDeadLine"),
        "office_address": opportunity.get("officeAddress"),
        "place_of_performance": opportunity.get("placeOfPerformance"),
        "point_of_contact": opportunity.get("pointOfContact"),
        "award": opportunity.get("award"),
    }
    prompt = f"""
You are a careful federal-contract capture analyst. Compare the opportunity with the
company profile and extract submission facts only when supported by the supplied record.
For an open opportunity without an official award amount, estimate a plausible total
contract-value range. Clearly base it on scope, duration, staffing, contract type, and
similar federal work; use null values and explain why when evidence is insufficient.
Never present place of performance as the electronic/mail submission destination unless
the record explicitly says it is.

Company profile: {profile}
Opportunity record: {json.dumps(prompt_record, default=str)}

Return only valid JSON with this shape:
{{
  "match_score": 0,
  "summary": "two concise sentences",
  "tech": ["technology"],
  "projected_value_low": null,
  "projected_value_high": null,
  "value_confidence": "low|medium|high",
  "value_basis": "short explanation",
  "submission_method": "portal|email|mail|in person|not published",
  "submission_location": "exact supported destination or Not published",
  "submission_instructions": "short supported instruction or Review solicitation attachments"
}}
"""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={gemini_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1},
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(clean_json_response(raw))
    except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        return {
            "match_score": 0,
            "summary": f"AI analysis unavailable: {exc}",
            "tech": [],
            "projected_value_low": None,
            "projected_value_high": None,
            "value_confidence": "low",
            "value_basis": "No estimate could be produced.",
            "submission_method": "not published",
            "submission_location": "Not published",
            "submission_instructions": "Review the solicitation and attachments on SAM.gov.",
        }


def official_award_amount(opportunity):
    award = opportunity.get("award")
    return award.get("amount") if isinstance(award, dict) else None


def display_opportunity(opportunity, analysis):
    score = analysis.get("match_score", 0)
    title = opportunity.get("title") or "Untitled opportunity"
    matched_naics = ", ".join(opportunity.get("_matched_naics", []))
    matched_psc = ", ".join(opportunity.get("_matched_psc", []))
    with st.expander(f"📊 Match: {score}% | {title}"):
        st.caption(
            f"Matched search NAICS: {matched_naics or 'None'} · "
            f"Matched search PSC: {matched_psc or 'None'} · "
            f"SAM.gov NAICS: {opportunity.get('naicsCode') or 'Not published'} · "
            f"SAM.gov PSC: {opportunity.get('classificationCode') or 'Not published'} · "
            f"Solicitation: {opportunity.get('solicitationNumber') or 'Not published'}"
        )
        summary_col, metric_col = st.columns([3, 1])
        with summary_col:
            st.write(f"**Agency:** {opportunity.get('fullParentPathName') or opportunity.get('department') or 'Unknown'}")
            st.write(f"**AI summary:** {analysis.get('summary', 'Not available')}")
            tech = analysis.get("tech") or []
            st.write(f"**Technologies:** {', '.join(tech) if isinstance(tech, list) else tech or 'None identified'}")
        with metric_col:
            st.metric("Match confidence", f"{score}%")
            st.link_button("Open solicitation", opportunity.get("uiLink") or "https://sam.gov/content/opportunities")

        st.markdown("#### Value")
        amount = official_award_amount(opportunity)
        if amount is not None:
            st.write(f"**Official award amount:** {currency(amount)}")
        else:
            low = analysis.get("projected_value_low")
            high = analysis.get("projected_value_high")
            estimate = f"{currency(low)} – {currency(high)}" if low is not None and high is not None else "Insufficient information"
            st.write(f"**AI projected contract value:** {estimate}")
            st.caption(
                f"Confidence: {analysis.get('value_confidence', 'low')}. "
                f"Basis: {analysis.get('value_basis', 'Not provided')} This is not an official government estimate."
            )

        deadline = opportunity.get("responseDeadLine") or "Not published"
        st.markdown("#### Location and submission")
        st.write(f"**Response deadline/time:** {deadline} (timezone as published by SAM.gov)")
        st.write(f"**Place of performance:** {format_address(opportunity.get('placeOfPerformance'))}")
        st.write(f"**Contracting office:** {format_address(opportunity.get('officeAddress'))}")
        st.write(f"**Submission method:** {analysis.get('submission_method', 'not published')}")
        st.write(f"**Submission destination:** {analysis.get('submission_location', 'Not published')}")
        st.write(f"**Instructions:** {analysis.get('submission_instructions', 'Review solicitation attachments')}")
        st.caption("Always verify the deadline, timezone, amendments, and delivery instructions in the official solicitation attachments.")


if st.button("🚀 Search & Analyze Contracts", type="primary"):
    naics_codes, invalid_naics = parse_naics_codes(naics_input)
    psc_codes, invalid_psc = parse_psc_codes(psc_input)
    if invalid_naics:
        st.error(f"Invalid NAICS entries: {', '.join(invalid_naics)}. Use 2-6 digits per code.")
    elif invalid_psc:
        st.error(f"Invalid PSC entries: {', '.join(invalid_psc)}. Use 1-4 letters/numbers per code.")
    elif not naics_codes and not psc_codes:
        st.error("Enter at least one NAICS or Product Service Code.")
    elif not sam_key or not llm_key:
        st.error("Enter both API keys in the sidebar.")
    else:
        search_count = len(naics_codes) + len(psc_codes)
        with st.spinner(f"Scanning SAM.gov for {search_count} search code(s)..."):
            opportunities, fetch_errors = fetch_contracts(
                sam_key, naics_codes, psc_codes, lookback_days, max_per_code
            )
        for error in fetch_errors:
            st.warning(error)

        if not opportunities:
            st.warning("No contracts were found for those NAICS codes and date range.")
        else:
            st.success(f"Found {len(opportunities)} unique contracts. Running AI analysis...")
            progress = st.progress(0)
            for index, opportunity in enumerate(opportunities, start=1):
                analysis = analyze_with_gemini(opportunity, llm_key, company_profile)
                display_opportunity(opportunity, analysis)
                progress.progress(index / len(opportunities))
