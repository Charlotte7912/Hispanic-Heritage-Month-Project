# Hispanic Heritage Historical Data Explorer + Function & Calculus Lab

A Streamlit classroom app for exploring official World Bank indicator data and studying mathematical functions, rates of change, and calculus concepts.

## Files
- `app.py` — self-contained Streamlit application
- `requirements.txt` — unpinned dependencies

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Visit https://share.streamlit.io and connect GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy and inspect logs if startup fails.
5. Test the app as a viewer, including a country/indicator change and each download.

## Deploy on Render
Create a Python Web Service connected to the repository. Leave Root Directory empty if `app.py` is in the repository root.

Build command:
```bash
pip install -r requirements.txt
```

Start command (one line):
```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

A free Render service may sleep after inactivity. Open the app before presenting.

## Data and definitions
The Historical Data Explorer queries the World Bank API for the last 70 calendar years ending in the previous year. The app displays actual available coverage and does not invent replacement observations. It uses the indicator definitions and units displayed in the app. Brazil is included as a Latin American comparison, not as a Spanish-speaking Hispanic country.

## Features
- World Bank indicator retrieval with timeout/error handling and caching
- Editable working table, row validation, and reset to source
- Centered-time polynomial fitting, in-sample R² and MAE, residual inspection
- Single-year and batch estimates
- Same-indicator country comparison, sample markers, forward display horizon, printer-friendly panel
- Genuine CSV, Excel workbook, and PDF report downloads
- Function lab with suggested/custom functions, symbolic derivatives, secant/tangent analysis, numerical extrema/inflection analysis, and context interpretation

## Known limitations
- Polynomial extrapolation may be implausible and is not a dependable forecast by itself.
- The app does not provide statistically validated prediction intervals.
- National unemployment estimates may not be directly comparable across countries.
- Calculus root classification includes numerical checks and may not resolve every expression.
- The simulation advances through user controls; it is not a continuous physical animation.
- No historical/cultural source research is embedded automatically. Students must add credible context sources, a research proposal, findings, and team contributions.

## Suggested research question
How have unemployment rates changed in Mexico and Colombia, and how well do polynomial models describe the observed trends?

## Team and AI-use disclosure
Complete this section with your team members, roles, contributions, and the ways you checked or revised AI-generated code. Verify the source data, calculations, and deployed app before submitting.
