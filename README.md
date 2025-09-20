BioScreener is a lightweight Python tool that discovers small- and mid-cap biotechnology companies with recent signs of drug-development activity.  
It combines public data sources with an optional OpenAI summary.

## Features
- News and Announcements  
  * Region-aware Google News RSS  
    – AU-biased for `.AX` (ASX) tickers  
    – US-biased for all others  
  * ASX company announcements (primary source) for `.AX` tickers
- ClinicalTrials.gov (optional) – counts and analyses registered trials
- Activity scoring (phase-agnostic)  
  * detects signals such as first-patient dosing, IND clearance, readouts, Fast-Track status, partnerships and similar events
- Optional OpenAI blurb  
  * If an API key is supplied, BioScreener generates a short, neutral summary of each company’s current development activity.

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/MaxmilionCarr/BioScreener.git
cd BioScreener
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
