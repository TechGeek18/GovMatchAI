import streamlit as st
import requests
import json
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="GovMatch AI", page_icon="🏛️", layout="wide")

st.title("🏛️ GovMatch AI - Internal Pipeline")
st.write("Enter your API keys in the sidebar and click the button to scan SAM.gov and analyze IT opportunities.")

# --- UI Sidebar ---
st.sidebar.header("⚙️ Configuration")
sam_key = st.sidebar.text_input("SAM.gov API Key", type="password")
llm_key = st.sidebar.text_input("Gemini API Key", type="password")
naics_input = st.sidebar.text_input("Target NAICS (comma separated)", "541511, 541512, 541519")

st.sidebar.markdown("---")
company_profile = st.sidebar.text_area("Your Company Profile (For AI Matching)", 
    "We are an IT consulting firm specializing in cloud architecture (AWS/Azure), custom Python/React development, and Zero Trust cybersecurity.")

# --- Fetching & AI Logic ---
def fetch_contracts(api_key, naics_list):
    today = datetime.now()
    yesterday = today - timedelta(days=2) # Looking back 2 days
    posted_from = yesterday.strftime("%m/%d/%Y")
    posted_to = today.strftime("%m/%d/%Y")
    
    url = "https://api.sam.gov/opportunities/v2/search"
    results = []
    for naics in naics_list:
        params = {"api_key": api_key, "ncode": naics.strip(), "postedFrom": posted_from, "postedTo": posted_to, "limit": 10}
        resp = requests.get(url, params=params)
        if resp.status_code == 200 and "opportunitiesData" in resp.json():
            results.extend(resp.json()["opportunitiesData"])
    return results

def analyze_with_gemini(opp, gemini_key, profile):
    title = opp.get("title", "No Title")
    description = str(opp.get("description", ""))
    
    prompt = f"""
    Compare this contract against our company profile.
    Company Profile: {profile}
    Contract Title: {title}
    Description: {description}
    
    Return ONLY a valid JSON object with:
    "match_score" (0-100), "summary" (2 sentences of what they want), "tech" (technologies needed)
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    try:
        r = requests.post(url, json=payload)
        res = r.json()
        raw = res['candidates'][0]['content']['parts'][0]['text']
        return json.loads(raw)
    except:
        return {"match_score": 50, "summary": "AI could not parse requirements.", "tech": "N/A"}

# --- Main App Action ---
if st.button("🚀 Search & Analyze Contracts"):
    if not sam_key or not llm_key:
        st.error("Please enter both API keys in the sidebar.")
    else:
        with st.spinner("Fetching data from SAM.gov..."):
            naics_list = naics_input.split(",")
            opportunities = fetch_contracts(sam_key, naics_list)
        
        if not opportunities:
            st.warning("No new contracts found in the last 48 hours for those NAICS codes.")
        else:
            st.success(f"Found {len(opportunities)} contracts. Analyzing with AI...")
            
            for opp in opportunities:
                # Run the actual AI Analysis
                analysis = analyze_with_gemini(opp, llm_key, company_profile)
                
                # Show results in a clean card
                with st.expander(f"📊 Match: {analysis.get('match_score')}% | {opp.get('title')}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Agency:** {opp.get('department', 'Unknown')}")
                        st.write(f"**AI Summary:** {analysis.get('summary')}")
                        st.write(f"**Tech Mentioned:** {analysis.get('tech')}")
                    with col2:
                        st.metric(label="Match Confidence", value=f"{analysis.get('match_score')}%")
                        st.link_button("Go to Solicitation", opp.get("uiLink", "https://sam.gov"))