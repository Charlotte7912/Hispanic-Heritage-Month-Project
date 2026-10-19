import io
import math
import re
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

# ---------------------------------------------------------------------
# Hispanic Heritage Historical Data Explorer + Function & Calculus Lab
# ---------------------------------------------------------------------
st.set_page_config(page_title="Hispanic Heritage Data & Function Lab", page_icon="📈", layout="wide")

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU",
    "Uruguay": "URY",
}
INDICATORS = {
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN", "units": "years",
        "definition": "Expected years of life at birth, total population.",
        "minimum": 0, "maximum": 125,
    },
    "Unemployment rate (national estimate)": {
        "code": "SL.UEM.TOTL.NE.ZS", "units": "% of total labor force",
        "definition": "Unemployment as a share of the labor force, national estimate. National methods may differ across countries.",
        "minimum": 0, "maximum": 100,
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD", "units": "current US$ per person",
        "definition": "Gross domestic product divided by midyear population, in current US dollars; not average household income or wealth.",
        "minimum": 0, "maximum": None,
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN", "units": "births per 1,000 people",
        "definition": "Annual number of live births per 1,000 people; this is a rate, not a count of births.",
        "minimum": 0, "maximum": 100,
    },
    "Net migration": {
        "code": "SM.POP.NETM", "units": "people (net)",
        "definition": "Immigration minus emigration over the source period; a net measure, not the number leaving.",
        "minimum": None, "maximum": None,
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5", "units": "homicides per 100,000 people",
        "definition": "Intentional homicides per 100,000 people, as reported in the World Bank series.",
        "minimum": 0, "maximum": None,
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG", "units": "annual % change",
        "definition": "Annual percentage change in consumer prices.",
        "minimum": -100, "maximum": None,
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS", "units": "% of GDP",
        "definition": "Central government debt as a percentage of GDP; not all public-sector debt.",
        "minimum": 0, "maximum": None,
    },
    "Total reserves, including gold": {
        "code": "FI.RES.TOTL.CD", "units": "current US$",
        "definition": "Total reserves including gold, reported in current US dollars.",
        "minimum": 0, "maximum": None,
    },
    "External balance: goods and services": {
        "code": "NE.RSB.GNFS.ZS", "units": "% of GDP",
        "definition": "External balance on goods and services as a percentage of GDP; not the current account.",
        "minimum": None, "maximum": None,
    },
}
WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
CONTEXTS = {
    "Business": ("Revenue", "currency units", "Profit", "currency units", "Revenue, cost, or profit can be modeled over time or another input."),
    "Engineering": ("Time", "seconds", "Position", "meters", "Position functions can be differentiated to estimate velocity and acceleration."),
    "Medical": ("Time", "hours", "Concentration", "mg/L", "A concentration curve may describe a simplified treatment scenario, not medical advice."),
    "Sports science": ("Time", "seconds", "Distance", "meters", "Distance or position can help discuss speed and acceleration."),
    "Biology": ("Time", "days", "Population", "organisms", "Population models are simplified and depend on assumptions."),
    "Bioengineering": ("Time", "hours", "Signal", "arbitrary units", "A signal model can help inspect rates and curvature."),
    "Economic forecasting": ("Year", "years", "Indicator value", "indicator units", "Economic extrapolations are uncertain and should not be treated as reliable forecasts by default."),
    "Weather": ("Time", "hours", "Temperature", "°C", "A smooth curve is a mathematical approximation, not a weather forecast."),
    "Custom": ("x", "user-defined units", "f(x)", "user-defined units", "Describe your own context and variables."),
}
SUGGESTED_FUNCTIONS = {
    "Polynomial": "x**3 - 3*x",
    "Rational": "(x**2 + 1)/(x - 1)",
    "Exponential": "2*exp(0.25*x)",
    "Logarithmic": "log(x)",
    "Trigonometric": "sin(x)",
}

def world_bank_data(country_code, indicator_code, start_year, end_year):
    """Fetch official World Bank observations; return an empty dataframe on a clear failure."""
    url = WB_BASE.format(country=country_code, indicator=indicator_code)
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["Year", "Value"]), f"Unexpected API response from {url}"
        rows = []
        for item in payload[1]:
            if not isinstance(item, dict):
                continue
            year, value = item.get("date"), item.get("value")
            try:
                y = int(year)
                v = float(value) if value is not None else np.nan
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                rows.append({"Year": y, "Value": v})
        frame = pd.DataFrame(rows, columns=["Year", "Value"]).drop_duplicates("Year").sort_values("Year")
        if frame.empty:
            return frame, f"The World Bank returned no non-missing observations for {start_year}–{end_year}."
        return frame.reset_index(drop=True), None
    except requests.Timeout:
        return pd.DataFrame(columns=["Year", "Value"]), "The World Bank request timed out. Try again later."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["Year", "Value"]), f"Could not retrieve data from the World Bank: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["Year", "Value"]), f"Could not parse the World Bank response: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_world_bank_data(country_code, indicator_code, start_year, end_year):
    return world_bank_data(country_code, indicator_code, start_year, end_year)

def clean_working_data(frame, indicator_name):
    """Validate edited rows without silently altering values."""
    info = INDICATORS[indicator_name]
    errors = []
    accepted = []
    seen = set()
    for idx, row in frame.iterrows():
        try:
            year_num = float(row["Year"])
            value_num = float(row["Value"])
        except (ValueError, TypeError, KeyError):
            errors.append(f"Row {idx + 1}: year and value must be numeric.")
            continue
        if not np.isfinite(year_num) or not year_num.is_integer():
            errors.append(f"Row {idx + 1}: year must be a finite integer.")
            continue
        year = int(year_num)
        if year < 1800 or year > date.today().year:
            errors.append(f"Row {idx + 1}: year {year} is outside the allowed range 1800–{date.today().year}.")
            continue
        if year in seen:
            errors.append(f"Row {idx + 1}: duplicate year {year}; duplicate rows are rejected.")
            continue
        if not np.isfinite(value_num):
            errors.append(f"Row {idx + 1}: value must be finite.")
            continue
        if info["minimum"] is not None and value_num < info["minimum"]:
            errors.append(f"Row {idx + 1}: {indicator_name} cannot be below {info['minimum']}.")
            continue
        if info["maximum"] is not None and value_num > info["maximum"]:
            errors.append(f"Row {idx + 1}: {indicator_name} cannot exceed {info['maximum']}.")
            continue
        seen.add(year)
        accepted.append({"Year": year, "Value": value_num})
    result = pd.DataFrame(accepted, columns=["Year", "Value"]).sort_values("Year").reset_index(drop=True)
    return result, errors

def fit_polynomial(frame, degree):
    if len(frame) < degree + 1 or frame["Year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; only {frame['Year'].nunique()} are available.")
    base_year = int(frame["Year"].min())
    t = frame["Year"].to_numpy(dtype=float) - base_year
    y = frame["Value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    residuals = y - pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    return {"base": base_year, "coeff": coeff, "degree": degree, "pred": pred,
            "residuals": residuals, "r2": r2, "mae": mae, "mean": float(np.mean(y)),
            "min_year": int(frame["Year"].min()), "max_year": int(frame["Year"].max()),
            "n": len(frame)}

def predict(model, years):
    years = np.asarray(years, dtype=float)
    return np.polyval(model["coeff"], years - model["base"])

def equation_string(model, precision=7):
    pieces = []
    degree = model["degree"]
    for i, c in enumerate(model["coeff"]):
        power = degree - i
        if abs(c) < 1e-14:
            continue
        ctext = f"{c:.{precision}g}"
        term = ctext if power == 0 else (f"{ctext}·t" if power == 1 else f"{ctext}·t^{power}")
        pieces.append(("- " if c < 0 else "+ ") + term if pieces else ("-" if c < 0 else "") + term)
    return "f(t) = " + (" ".join(pieces) if pieces else "0") + f", where t = year − {model['base']}"

def classify_year(year, model, observed_years):
    if int(year) in observed_years:
        return "Fitted value at an observed year"
    if model["min_year"] <= year <= model["max_year"]:
        return "Within fitted range (interpolation)"
    return "Outside fitted range (extrapolation)"

def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")

def excel_bytes(summary, data, results=None):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working Data", index=False)
        if results is not None and not results.empty:
            results.to_excel(writer, sheet_name="Calculated Years", index=False)
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, 20, 20)
    return buffer.getvalue()

def pdf_bytes(title, summary_lines, data, results=None, chart_png=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=0.45*inch,
                           leftMargin=0.45*inch, topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in summary_lines:
        story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
    story.append(Spacer(1, 8))
    if chart_png:
        story.append(RLImage(io.BytesIO(chart_png), width=8.7*inch, height=4.3*inch))
        story.append(Spacer(1, 8))
    story.append(Paragraph("Working data preview (first 30 rows)", styles["Heading2"]))
    subset = data.head(30)
    table_data = [list(subset.columns)] + subset.round(5).astype(str).values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f3")]),
    ]))
    story.append(table)
    if results is not None and not results.empty:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Calculated-year results", styles["Heading2"]))
        rdata = [list(results.columns)] + results.round(5).astype(str).values.tolist()
        rtable = Table(rdata, repeatRows=1)
        rtable.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#263238")),
                                    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                    ("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(rtable)
    doc.build(story)
    return buffer.getvalue()

def parse_function(expr_text, x_symbol):
    """Parse common math notation safely with SymPy; only known math symbols/functions."""
    normalized = expr_text.replace("^", "**").replace("×", "*").replace("÷", "/")
    allowed = {"x": x_symbol, "pi": sp.pi, "e": sp.E, "sin": sp.sin, "cos": sp.cos,
               "tan": sp.tan, "exp": sp.exp, "log": sp.log, "ln": sp.log,
               "sqrt": sp.sqrt, "abs": sp.Abs}
    try:
        expr = sp.sympify(normalized, locals=allowed, evaluate=True)
        if expr.has(sp.Function) and any(str(node.func) not in
            {"sin", "cos", "tan", "exp", "log", "Abs", "sqrt"} for node in sp.preorder_traversal(expr) if isinstance(node, sp.Function)):
            raise ValueError("Use only +, -, *, /, powers, sin, cos, tan, exp, log/ln, sqrt, and abs.")
        if not expr.free_symbols.issubset({x_symbol}):
            raise ValueError("The function may contain only the variable x.")
        return expr
    except Exception as exc:
        raise ValueError(f"Could not read that function. Use notation such as x^3 - 3*x, exp(x), log(x), or sin(x). Details: {exc}")

def numeric_function(expr, x_symbol):
    fn = sp.lambdify(x_symbol, expr, modules=["numpy"])
    def evaluate(values):
        arr = np.asarray(values, dtype=float)
        with np.errstate(all="ignore"):
            result = fn(arr)
        result = np.asarray(result, dtype=float)
        if result.shape == ():
            result = np.full(arr.shape, float(result))
        return result
    return evaluate

def safe_float(value, fallback=0.0):
    try:
        v = float(value)
        return v if np.isfinite(v) else fallback
    except (ValueError, TypeError):
        return fallback

# Sidebar navigation
st.sidebar.title("Hispanic Heritage Data Lab")
page = st.sidebar.radio("Choose a lab", ["Historical Data Explorer", "Function & Calculus Lab"])
st.sidebar.caption("A student learning tool. Models describe data; they do not prove causes or guarantee forecasts.")

if page == "Historical Data Explorer":
    st.title("📈 Hispanic Heritage Historical Data Explorer")
    st.write("Turn official country indicators into mathematical investigations. The app retrieves World Bank observations, lets you edit a working copy, fits a polynomial, and exports a report.")
    st.info("Brazil is included as a Latin American comparison; it is not a Spanish-speaking Hispanic country. Center your showcase on at least two Spanish-speaking countries.")
    with st.sidebar:
        st.header("Data and model")
        country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
        indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()), index=1)
        degree = st.slider("Polynomial degree", 3, 8, 3)
        sample_step = st.slider("Model marker / sample increment (years)", 1, 10, 1)
        horizon = st.slider("Forward display horizon (years)", 0, 50, 5)
        comparison_names = st.multiselect("Compare same indicator with", [c for c in COUNTRIES if c != country_name], default=["Colombia"] if country_name != "Colombia" else ["Mexico"])
        printer_view = st.toggle("Printer-friendly view", value=False)
    current_year = date.today().year
    start_year, end_year = current_year - 70, current_year - 1
    info = INDICATORS[indicator_name]
    st.caption(f"Search window: {start_year}–{end_year}. Actual coverage and observation count depend on available official data.")
    with st.spinner("Retrieving official World Bank observations…"):
        source_data, fetch_error = cached_world_bank_data(COUNTRIES[country_name], info["code"], start_year, end_year)
    source_data = source_data.copy()
    if fetch_error:
        st.error(fetch_error)
        st.warning("Live retrieval is unavailable. No synthetic replacement data are generated. Check your connection or try again later.")
    st.markdown(f"**Definition:** {info['definition']}")
    st.markdown(f"**Source:** [World Bank indicator {info['code']}](https://data.worldbank.org/indicator/{info['code']}) · [API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries)")
    st.caption(f"Retrieved on {datetime.now().strftime('%Y-%m-%d %H:%M')} · {country_name} · {info['units']} · "
               + (f"observed coverage {int(source_data['Year'].min())}–{int(source_data['Year'].max())}, n={len(source_data)}" if not source_data.empty else "no usable observations"))
    if "source_frame_key" not in st.session_state or st.session_state.source_frame_key != f"{country_name}|{indicator_name}|{start_year}|{end_year}":
        st.session_state.source_frame_key = f"{country_name}|{indicator_name}|{start_year}|{end_year}"
        st.session_state.source_frame = source_data.copy()
        st.session_state.working_frame = source_data.copy()
        st.session_state.data_edited = False
        st.session_state.last_single = None
        st.session_state.batch_results = pd.DataFrame()
    if st.button("Reset working table to source", disabled=source_data.empty):
        st.session_state.working_frame = st.session_state.source_frame.copy()
        st.session_state.data_edited = False
        st.session_state.last_single = None
        st.session_state.batch_results = pd.DataFrame()
        st.rerun()
    st.subheader("1. Inspect and edit the working data")
    st.write("Edit Year or Value in the table, then use **Apply and validate edits**. The original source observations remain separate.")
    if source_data.empty:
        st.warning("There is no source table to edit yet.")
    else:
        editor_key = f"editor_{country_name}_{info['code']}"
        edited_table = st.data_editor(st.session_state.working_frame, num_rows="dynamic", use_container_width=True,
                                      key=editor_key, column_config={
                                          "Year": st.column_config.NumberColumn("Year", step=1, format="%d"),
                                          "Value": st.column_config.NumberColumn(f"Value ({info['units']})", format="%.8g"),
                                      })
        if st.button("Apply and validate edits"):
            valid_frame, validation_errors = clean_working_data(edited_table, indicator_name)
            st.session_state.working_frame = valid_frame
            st.session_state.data_edited = not valid_frame.equals(st.session_state.source_frame.reset_index(drop=True))
            st.session_state.last_single = None
            st.session_state.batch_results = pd.DataFrame()
            if validation_errors:
                st.warning("Some rows were rejected:\n\n" + "\n".join(validation_errors[:20]))
            st.success(f"Accepted {len(valid_frame)} valid rows. " + ("This is labeled a user-modified scenario." if st.session_state.data_edited else "Working data match the source."))
            st.rerun()
    working = st.session_state.get("working_frame", source_data).copy()
    if working.empty:
        st.stop()
    # Revalidate stored working table, and ensure year/value types are numeric.
    working, validation_errors = clean_working_data(working, indicator_name)
    if validation_errors:
        st.warning("Working data validation: " + "; ".join(validation_errors[:5]))
    if len(working) < degree + 1:
        st.error(f"Not enough distinct valid observations to fit degree {degree}. Need at least {degree + 1}; currently {len(working)}.")
        st.stop()
    try:
        model = fit_polynomial(working, degree)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))
        st.stop()
    title_prefix = "USER-MODIFIED SCENARIO — " if st.session_state.get("data_edited", False) else ""
    st.subheader("2. Polynomial model")
    st.markdown(f"**{title_prefix}{country_name}: {indicator_name}**")
    st.code(equation_string(model))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observations", model["n"])
    c2.metric("Mean", f"{model['mean']:.5g}")
    c3.metric("In-sample R²", f"{model['r2']:.5f}" if np.isfinite(model["r2"]) else "undefined")
    c4.metric("In-sample MAE", f"{model['mae']:.5g}")
    c5.metric("Fitted range", f"{model['min_year']}–{model['max_year']}")
    st.caption(f"Base year = {model['base']}; t = year − {model['base']}. R² and MAE are in-sample fit metrics, not proof of forecasting accuracy.")
    # Graph
    st.subheader("3. Graph and same-indicator comparison")
    chart_frame = working.copy()
    x_min = int(working["Year"].min())
    x_max = int(working["Year"].max())
    display_max = min(2100, x_max + horizon)
    smooth_x = np.linspace(x_min, max(x_min + 1, display_max), max(300, (display_max-x_min+1)*2))
    smooth_y = predict(model, smooth_x)
    sampled_x = np.arange(x_min, display_max + 1, sample_step)
    sampled_y = predict(model, sampled_x)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.scatter(working["Year"], working["Value"], label="Observed / working data", s=24, zorder=3)
    ax.plot(smooth_x[smooth_x <= x_max], smooth_y[smooth_x <= x_max], label="Fitted curve (historical range)", linewidth=2)
    if display_max > x_max:
        ax.plot(smooth_x[smooth_x > x_max], smooth_y[smooth_x > x_max], "--", label="Extrapolated display segment", linewidth=2)
        ax.axvline(x_max, linestyle=":", linewidth=1.5, label="End of historical range")
    ax.scatter(sampled_x, sampled_y, marker="x", s=28, label=f"Sampled model markers (every {sample_step} year(s))")
    for compare_name in comparison_names:
        with st.spinner(f"Retrieving comparison data for {compare_name}…"):
            comp_data, comp_error = cached_world_bank_data(COUNTRIES[compare_name], info["code"], start_year, end_year)
        if comp_error:
            st.warning(f"{compare_name}: {comp_error}")
        elif not comp_data.empty:
            ax.plot(comp_data["Year"], comp_data["Value"], marker=".", linewidth=1.3, markersize=3,
                    label=f"{compare_name} observed ({comp_data['Year'].min()}–{comp_data['Year'].max()}, n={len(comp_data)})")
        else:
            st.info(f"No non-missing comparison data were returned for {compare_name}; no substitute series was plotted.")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({info['units']})")
    ax.set_title(f"{country_name}: {indicator_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    if printer_view:
        st.markdown("---")
        st.subheader("Printer-friendly summary")
        st.write(f"**Research prompt:** How has {indicator_name.lower()} changed in {country_name}, and how does its pattern compare with selected countries?")
        st.write(f"**Model:** {equation_string(model)}")
        st.write(f"**Coverage:** {model['min_year']}–{model['max_year']} ({model['n']} observations). **MAE:** {model['mae']:.5g}.")
        st.write("**Limitations:** Missing years may reflect unavailable official data. Polynomial fits can behave implausibly outside the observed range. No statistically validated prediction intervals are provided. Correlation and trend do not establish causation.")
    st.subheader("4. Calculate estimates")
    with st.form("single_estimate_form"):
        single_year = st.number_input("Year to estimate (1950–2100)", min_value=1950, max_value=2100, value=min(2100, max(1950, x_max + 1)), step=1)
        calculate_single = st.form_submit_button("Calculate value")
    observed_years = set(working["Year"].astype(int).tolist())
    if calculate_single:
        estimate = float(predict(model, [single_year])[0])
        actual_rows = working.loc[working["Year"] == int(single_year), "Value"]
        st.session_state.last_single = {"Year": int(single_year), "Estimate": estimate,
                                        "Units": info["units"], "Classification": classify_year(single_year, model, observed_years),
                                        "Observed value": float(actual_rows.iloc[0]) if not actual_rows.empty else np.nan}
    last_single = st.session_state.get("last_single")
    if last_single:
        st.write(f"**{last_single['Year']}:** {last_single['Estimate']:.8g} {last_single['Units']} — {last_single['Classification']}.")
        if np.isfinite(last_single["Observed value"]):
            st.write(f"Observed value at that year: {last_single['Observed value']:.8g} {last_single['Units']}. The model estimate is not a replacement for the observation.")
        if last_single["Year"] < x_min or last_single["Year"] > x_max:
            st.warning("This is extrapolation. Polynomial behavior can become implausible outside the fitted range; values are not clipped.")
    with st.form("batch_estimate_form"):
        batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            bc1, bc2, bc3 = st.columns(3)
            batch_start = bc1.number_input("Start year", 1950, 2100, max(1950, x_min), step=1)
            batch_end = bc2.number_input("End year", 1950, 2100, min(2100, max(x_min + 5, x_max)), step=1)
            batch_step = bc3.number_input("Step (1–10 years)", 1, 10, 1, step=1)
            custom_text = ""
        else:
            custom_text = st.text_input("Comma-separated years", value=f"{x_min},{x_max},{min(2100,x_max+5)}")
            batch_start, batch_end, batch_step = 0, -1, 1
        calculate_batch = st.form_submit_button("Calculate batch")
    if calculate_batch:
        valid_years, issues = [], []
        if batch_mode == "Year range":
            if batch_end <= batch_start:
                issues.append("End year must be greater than start year.")
            else:
                valid_years = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
        else:
            for token in custom_text.split(","):
                token = token.strip()
                if not token:
                    continue
                if not re.fullmatch(r"[+-]?\d+", token):
                    issues.append(f"'{token}' is not an integer year.")
                    continue
                year = int(token)
                if year < 1950 or year > 2100:
                    issues.append(f"{year} is outside 1950–2100.")
                    continue
                if year in valid_years:
                    issues.append(f"Duplicate year {year} was ignored.")
                    continue
                valid_years.append(year)
        if issues:
            st.warning("Input feedback: " + " ".join(issues))
        valid_years = sorted(set(valid_years))
        if not valid_years:
            st.error("No valid years to calculate.")
            st.session_state.batch_results = pd.DataFrame()
        else:
            vals = predict(model, valid_years)
            st.session_state.batch_results = pd.DataFrame({
                "Year": valid_years, "Estimate": vals, "Units": info["units"],
                "Classification": [classify_year(y, model, observed_years) for y in valid_years],
            })
            st.success(f"Calculated {len(valid_years)} year(s): {', '.join(map(str, valid_years))}")
    results = st.session_state.get("batch_results", pd.DataFrame())
    if not results.empty:
        st.dataframe(results, use_container_width=True)
        outside = results[results["Classification"].str.contains("extrapolation", case=False)]
        if not outside.empty:
            st.warning("Some batch values are extrapolations. Treat them cautiously; no validated prediction intervals are provided.")
    st.subheader("5. Residuals and model limitations")
    residual_frame = working.copy()
    residual_frame["Fitted value"] = model["pred"]
    residual_frame["Residual (observed − fitted)"] = model["residuals"]
    with st.expander("Inspect residuals"):
        st.dataframe(residual_frame, use_container_width=True)
        st.write(f"Residual MAE = {model['mae']:.6g}. Residual error does not quantify all future uncertainty. This app does not provide statistically validated 95% prediction intervals.")
    st.markdown("**Research reminders:** compare at least two polynomial degrees; compute two average rates of change; use a chronological holdout if there are enough observations; write three supported findings, two limitations, and a follow-up question. Use historical/cultural sources to explain context without claiming the indicator alone defines a country.")
    st.markdown("**Data status:** " + ("User-modified scenario" if st.session_state.get("data_edited", False) else "Source data / unmodified working copy"))
    # Export
    st.subheader("6. Download reports")
    summary = {
        "Country": country_name, "Indicator": indicator_name, "Indicator code": info["code"],
        "Units": info["units"], "Definition": info["definition"],
        "Source": f"https://data.worldbank.org/indicator/{info['code']}",
        "Retrieved at": datetime.now().isoformat(timespec="minutes"),
        "Observed coverage": f"{x_min}–{x_max}", "Observation count": len(working),
        "Degree": degree, "Base year": model["base"], "Equation": equation_string(model),
        "In-sample R-squared": model["r2"], "In-sample MAE": model["mae"],
        "Data status": "user-modified scenario" if st.session_state.get("data_edited", False) else "source/unmodified",
        "Limitations": "No validated prediction intervals; polynomial extrapolation can be implausible.",
    }
    export_data = working.copy()
    export_data["Data status"] = summary["Data status"]
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
    chart_png = chart_buffer.getvalue()
    pdf_summary = [f"{k}: {v}" for k, v in summary.items()]
    pdf_summary += ["Findings/limitations: This fit describes the selected working observations; it does not establish causation. Missing data and cross-country definitions may limit comparisons. Extrapolations are uncertain."]
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download CSV", data=csv_bytes(export_data), file_name=f"{country_name}_{info['code']}_working_data.csv", mime="text/csv")
    e2.download_button("Download Excel (.xlsx)", data=excel_bytes(summary, export_data, results), file_name=f"{country_name}_{info['code']}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    e3.download_button("Download PDF report", data=pdf_bytes("Hispanic Heritage Historical Data Explorer", pdf_summary, export_data, results, chart_png), file_name=f"{country_name}_{info['code']}_report.pdf", mime="application/pdf")
    st.caption("The CSV contains the cleaned working table and edit status. Excel includes summary, working data, and calculated years when present. PDF includes metadata, model summary, chart, data preview, and calculated results.")

else:
    st.title("🧮 Function & Calculus Lab")
    st.write("Explore functions in contexts from business to engineering, biology, medicine, sports science, weather, and more. This is a classroom model—not professional advice.")
    mode = st.sidebar.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
    context = st.sidebar.selectbox("Context", list(CONTEXTS.keys()))
    x_name_default, x_unit_default, y_name_default, y_unit_default, context_help = CONTEXTS[context]
    st.sidebar.caption(context_help)
    x_name = st.sidebar.text_input("Independent variable name", x_name_default)
    x_unit = st.sidebar.text_input("Independent variable units", x_unit_default)
    y_name = st.sidebar.text_input("Dependent variable name", y_name_default)
    y_unit = st.sidebar.text_input("Dependent variable units", y_unit_default)
    context_description = st.sidebar.text_area("Context description", context_help)
    function_source = st.radio("Choose function source", ["Suggested function", "Build from parameters", "Enter my own function", "Use historical-data best-fit curve"], horizontal=True)
    if function_source == "Suggested function":
        function_type = st.selectbox("Function family", list(SUGGESTED_FUNCTIONS.keys()))
        expr_text = SUGGESTED_FUNCTIONS[function_type]
        st.code(f"f(x) = {expr_text}")
        st.caption("The suggested examples are mathematical demonstrations. Choose a domain that is appropriate for the expression.")
    elif function_source == "Build from parameters":
        family = st.selectbox("Function family", ["Polynomial", "Exponential", "Logarithmic", "Trigonometric", "Rational"])
        if family == "Polynomial":
            a = st.number_input("a (cubic coefficient)", value=1.0, step=0.5)
            b = st.number_input("b (quadratic coefficient)", value=0.0, step=0.5)
            c = st.number_input("c (linear coefficient)", value=-3.0, step=0.5)
            d = st.number_input("d (constant)", value=0.0, step=0.5)
            expr_text = f"({a})*x^3 + ({b})*x^2 + ({c})*x + ({d})"
        elif family == "Exponential":
            a = st.number_input("Scale a", value=2.0, step=0.5)
            k = st.number_input("Growth rate k", value=0.25, step=0.05)
            expr_text = f"({a})*exp(({k})*x)"
        elif family == "Logarithmic":
            a = st.number_input("Scale a", value=1.0, step=0.5)
            shift = st.number_input("Horizontal shift h", value=0.0, step=0.5)
            expr_text = f"({a})*log(x-({shift}))"
        elif family == "Trigonometric":
            a = st.number_input("Amplitude a", value=1.0, step=0.5)
            k = st.number_input("Frequency k", value=1.0, step=0.25)
            expr_text = f"({a})*sin(({k})*x)"
        else:
            a = st.number_input("Numerator scale a", value=1.0, step=0.5)
            shift = st.number_input("Denominator shift h", value=1.0, step=0.5)
            expr_text = f"(x^2+({a}))/(x-({shift}))"
        st.code(f"f(x) = {expr_text}")
    elif function_source == "Enter my own function":
        expr_text = st.text_input("Enter f(x)", value="x^3 - 3*x", help="Use x, + - * /, powers (^ or **), sin, cos, tan, exp, log/ln, sqrt, abs, and parentheses.")
    else:
        st.info("To use a historical-data best-fit curve, open the Historical Data Explorer first. This lab keeps its examples independent so the mathematical function can be changed safely.")
        expr_text = st.text_input("Enter a polynomial fitted-curve approximation", value="x^3 - 3*x")
    with st.expander("Set analysis interval and point"):
        c1, c2, c3 = st.columns(3)
        x_min_user = c1.number_input("Interval start", value=-4.0, step=0.5)
        x_max_user = c2.number_input("Interval end", value=4.0, step=0.5)
        x_point = c3.number_input("Point x₀", value=1.0, step=0.25)
        c4, c5 = st.columns(2)
        show_f = c4.checkbox("Show f(x)", value=True)
        show_d1 = c4.checkbox("Show f′(x)", value=True)
        show_d2 = c5.checkbox("Show f″(x)", value=(mode == "Calculus"))
    if x_max_user <= x_min_user:
        st.error("The interval end must be greater than the interval start.")
        st.stop()
    x = sp.Symbol("x", real=True)
    try:
        expr = parse_function(expr_text, x)
        d1 = sp.diff(expr, x)
        d2 = sp.diff(d1, x)
        f_eval, d1_eval, d2_eval = numeric_function(expr, x), numeric_function(d1, x), numeric_function(d2, x)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    st.markdown(f"**Context:** {context_description}")
    st.markdown(f"**Function:** \(f(x) = {sp.latex(expr)}\)")
    if mode == "Calculus":
        st.markdown(f"**First derivative:** \(f'(x) = {sp.latex(d1)}\)")
        st.markdown(f"**Second derivative:** \(f''(x) = {sp.latex(d2)}\)")
    elif mode == "Precalculus":
        st.markdown("Precalculus mode emphasizes graph shape, average rates of change, turning behavior, and how coefficients affect the model.")
    else:
        st.markdown("Honors Algebra 2 mode emphasizes domain/range, end behavior, graph trends, and average rates of change.")
    xs = np.linspace(x_min_user, x_max_user, 700)
    ys, dys, ddys = f_eval(xs), d1_eval(xs), d2_eval(xs)
    valid = np.isfinite(ys)
    # Break plot at undefined/non-finite points instead of connecting through them.
    fig, ax = plt.subplots(figsize=(10, 5.3))
    if show_f:
        ax.plot(xs, np.where(valid, ys, np.nan), label="f(x)", linewidth=2.2)
    if show_d1:
        ax.plot(xs, np.where(np.isfinite(dys), dys, np.nan), label="f′(x)", linewidth=1.5)
    if show_d2:
        ax.plot(xs, np.where(np.isfinite(ddys), ddys, np.nan), label="f″(x)", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(x_point, linestyle=":", linewidth=1, label=f"x₀ = {x_point:g}")
    ax.set_xlabel(f"{x_name} ({x_unit})")
    ax.set_ylabel(f"Value / rate (units vary by curve)")
    ax.set_title(f"{context}: function and derivatives")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Each derivative curve has different units: f(x) uses dependent-variable units; f′ uses dependent units per independent unit; f″ uses dependent units per independent unit squared.")
    # Average rate / secant
    st.subheader("Rates of change")
    left, right = st.columns(2)
    with left:
        aroc_a = st.number_input("Average-rate interval start a", value=float(x_min_user), key="aroc_a")
        aroc_b = st.number_input("Average-rate interval end b", value=float(x_max_user), key="aroc_b")
        if aroc_a == aroc_b:
            st.warning("Choose different endpoints for average rate of change.")
        else:
            fa, fb = safe_float(f_eval([aroc_a])[0], np.nan), safe_float(f_eval([aroc_b])[0], np.nan)
            if np.isfinite(fa) and np.isfinite(fb):
                average_rate = (fb-fa)/(aroc_b-aroc_a)
                st.metric("Average rate of change", f"{average_rate:.7g} {y_unit}/{x_unit}")
                st.write(f"Secant line: y = {fa:.6g} + ({average_rate:.6g})(x − {aroc_a:g})")
                secant_x = np.linspace(min(aroc_a, aroc_b), max(aroc_a, aroc_b), 100)
                secant_y = fa + average_rate*(secant_x-aroc_a)
                fig_sec, ax_sec = plt.subplots(figsize=(6.5, 3.2))
                ax_sec.plot(xs, np.where(valid, ys, np.nan), label="f(x)")
                ax_sec.plot(secant_x, secant_y, "--", label="Secant line")
                ax_sec.scatter([aroc_a, aroc_b], [fa, fb], zorder=4)
                ax_sec.legend()
                ax_sec.grid(True, alpha=0.2)
                st.pyplot(fig_sec, use_container_width=True)
            else:
                st.warning("At least one endpoint is outside the function's real-valued domain.")
    with right:
        f0 = safe_float(f_eval([x_point])[0], np.nan)
        slope0 = safe_float(d1_eval([x_point])[0], np.nan)
        if np.isfinite(f0):
            st.metric("Function value", f"{f0:.7g} {y_unit}")
        else:
            st.warning("The function is undefined or non-real at the selected point.")
        if np.isfinite(slope0) and np.isfinite(f0):
            st.metric("Instantaneous rate of change", f"{slope0:.7g} {y_unit}/{x_unit}")
            tangent_x = np.linspace(x_min_user, x_max_user, 250)
            tangent_y = f0 + slope0*(tangent_x-x_point)
            fig_tan, ax_tan = plt.subplots(figsize=(6.5, 3.2))
            ax_tan.plot(xs, np.where(valid, ys, np.nan), label="f(x)")
            ax_tan.plot(tangent_x, tangent_y, "--", label="Tangent at x₀")
            ax_tan.scatter([x_point], [f0], zorder=4)
            ax_tan.legend()
            ax_tan.grid(True, alpha=0.2)
            st.pyplot(fig_tan, use_container_width=True)
        else:
            st.warning("The derivative or function is not finite at the selected point, so a tangent line is not displayed.")
    # Contextual interpretation
    st.subheader("Interpretation in context")
    if np.isfinite(f0) and np.isfinite(slope0):
        trend = "increasing locally" if slope0 > 1e-9 else ("decreasing locally" if slope0 < -1e-9 else "momentarily flat")
        st.write(f"At x = {x_point:g}, the modeled {y_name.lower()} is about {f0:.6g} {y_unit} and is {trend}. Its instantaneous rate is {slope0:.6g} {y_unit} per {x_unit}.")
        if mode == "Calculus" and np.isfinite(safe_float(d2_eval([x_point])[0], np.nan)):
            second0 = safe_float(d2_eval([x_point])[0], np.nan)
            curvature = "concave up locally" if second0 > 1e-9 else ("concave down locally" if second0 < -1e-9 else "has approximately zero second derivative at this point")
            st.write(f"The second derivative is {second0:.6g} {y_unit}/{x_unit}², so the graph is {curvature}.")
    # Numerical / symbolic analysis
    st.subheader("Function analysis")
    try:
        domain = sp.calculus.util.continuous_domain(expr, x, sp.Interval(float(x_min_user), float(x_max_user)))
        st.write(f"Real-valued continuous domain on the selected interval: `{domain}`")
    except Exception:
        st.write("Domain: inspect the formula and graph for restrictions (for example, log arguments must be positive and denominators cannot be zero).")
    if mode in ["Precalculus", "Calculus"]:
        try:
            critical_expr = sp.solve(sp.Eq(d1, 0), x)
            critical_rows = []
            for root in critical_expr:
                try:
                    rv = float(sp.N(root))
                    if x_min_user <= rv <= x_max_user and abs(float(sp.im(sp.N(root)))) < 1e-8:
                        fval = safe_float(f_eval([rv])[0], np.nan)
                        left_slope = safe_float(d1_eval([rv-1e-4])[0], np.nan)
                        right_slope = safe_float(d1_eval([rv+1e-4])[0], np.nan)
                        if left_slope > 0 and right_slope < 0:
                            kind = "local maximum (numerical derivative test)"
                        elif left_slope < 0 and right_slope > 0:
                            kind = "local minimum (numerical derivative test)"
                        else:
                            kind = "stationary; no local extremum confirmed"
                        critical_rows.append({"x": rv, "f(x)": fval, "classification": kind})
                except Exception:
                    continue
            if critical_rows:
                st.write("Critical points in the interval:")
                st.dataframe(pd.DataFrame(critical_rows), use_container_width=True)
            else:
                st.write("No real critical points were found in the selected interval by the symbolic solver (or roots could not be resolved).")
        except Exception as exc:
            st.write(f"Critical-point solving was not available for this expression: {exc}")
    if mode == "Calculus":
        try:
            second_roots = sp.solve(sp.Eq(d2, 0), x)
            inflection_rows = []
            for root in second_roots:
                try:
                    rv = float(sp.N(root))
                    if x_min_user < rv < x_max_user and abs(float(sp.im(sp.N(root)))) < 1e-8:
                        before = safe_float(d2_eval([rv-1e-4])[0], np.nan)
                        after = safe_float(d2_eval([rv+1e-4])[0], np.nan)
                        if np.isfinite(before) and np.isfinite(after) and before*after < 0:
                            inflection_rows.append({"x": rv, "f(x)": safe_float(f_eval([rv])[0], np.nan), "concavity change": "confirmed numerically"})
                except Exception:
                    continue
            if inflection_rows:
                st.write("Inflection points (concavity change checked numerically):")
                st.dataframe(pd.DataFrame(inflection_rows), use_container_width=True)
            else:
                st.write("No inflection point with a numerically confirmed concavity change was found in the selected interval.")
            # Closed-interval extrema including endpoints, evaluated numerically.
            candidates = [float(x_min_user), float(x_max_user)]
            try:
                for root in sp.solve(sp.Eq(d1, 0), x):
                    rv = float(sp.N(root))
                    if x_min_user < rv < x_max_user and abs(float(sp.im(sp.N(root)))) < 1e-8:
                        candidates.append(rv)
            except Exception:
                pass
            evaluated = [(z, safe_float(f_eval([z])[0], np.nan)) for z in candidates]
            evaluated = [(z, val) for z, val in evaluated if np.isfinite(val)]
            if evaluated:
                lo = min(evaluated, key=lambda p:p[1])
                hi = max(evaluated, key=lambda p:p[1])
                st.write(f"Numerical absolute minimum on the selected closed interval: x ≈ {lo[0]:.6g}, f(x) ≈ {lo[1]:.6g}.")
                st.write(f"Numerical absolute maximum on the selected closed interval: x ≈ {hi[0]:.6g}, f(x) ≈ {hi[1]:.6g}.")
        except Exception as exc:
            st.write(f"Some calculus analysis could not be completed for this expression: {exc}")
    st.subheader("Interactive point simulation")
    st.write("Move the point slider to synchronize the point value, tangent slope, and contextual explanation. The slider is the dependable classroom control; use the play/pause controls to step through a short sequence.")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    sim_start = sim_col1.number_input("Simulation start", value=float(x_min_user), key="sim_start")
    sim_end = sim_col2.number_input("Simulation end", value=float(x_max_user), key="sim_end")
    sim_speed = sim_col3.select_slider("Playback step size", options=[0.05, 0.1, 0.25, 0.5, 1.0], value=0.25)
    if sim_end <= sim_start:
        st.warning("Simulation end must be greater than simulation start.")
    else:
        if "sim_x" not in st.session_state or st.session_state.get("sim_bounds") != (sim_start, sim_end):
            st.session_state.sim_x = min(max(float(x_point), sim_start), sim_end)
            st.session_state.sim_bounds = (sim_start, sim_end)
        s1, s2, s3 = st.columns(3)
        if s1.button("▶ Step / play once"):
            st.session_state.sim_x = min(sim_end, st.session_state.sim_x + sim_speed)
        if s2.button("⏸ Pause"):
            st.session_state.sim_paused = True
        if s3.button("↺ Reset"):
            st.session_state.sim_x = sim_start
        sim_x = st.slider("Current simulation x", float(sim_start), float(sim_end),
                          value=float(min(max(st.session_state.sim_x, sim_start), sim_end)), step=float(sim_speed), key="sim_slider")
        st.session_state.sim_x = sim_x
        sim_y = safe_float(f_eval([sim_x])[0], np.nan)
        sim_slope = safe_float(d1_eval([sim_x])[0], np.nan)
        sim_second = safe_float(d2_eval([sim_x])[0], np.nan)
        if np.isfinite(sim_y):
            st.write(f"**Moving point:** ({sim_x:.4g}, {sim_y:.4g}). **Tangent slope:** {sim_slope:.4g} {y_unit}/{x_unit}" if np.isfinite(sim_slope) else f"**Moving point:** ({sim_x:.4g}, {sim_y:.4g}); tangent unavailable.")
            sim_fig, sim_ax = plt.subplots(figsize=(9, 3.8))
            sim_ax.plot(xs, np.where(valid, ys, np.nan), label="f(x)")
            sim_ax.scatter([sim_x], [sim_y], s=55, label="Current point", zorder=5)
            if np.isfinite(sim_slope):
                tx = np.linspace(max(sim_start, sim_x-(sim_end-sim_start)*0.18), min(sim_end, sim_x+(sim_end-sim_start)*0.18), 80)
                sim_ax.plot(tx, sim_y + sim_slope*(tx-sim_x), "--", label="Tangent")
            sim_ax.legend()
            sim_ax.grid(True, alpha=0.2)
            st.pyplot(sim_fig, use_container_width=True)
            if context == "Engineering" or context == "Sports science":
                st.write(f"Interpretation: modeled position/distance is {sim_y:.4g} {y_unit}; rate is {sim_slope:.4g} {y_unit}/{x_unit}; acceleration-like second derivative is {sim_second:.4g} {y_unit}/{x_unit}².")
            elif context == "Medical":
                st.write("Concentration curve is an illustrative model only; it must not be used to change a real dose or treatment.")
            elif context == "Weather":
                st.write("This mathematical curve does not replace measured weather observations or an official forecast.")
            else:
                st.write(f"At this input, the modeled {y_name.lower()} is {sim_y:.4g} {y_unit}; the local rate is {sim_slope:.4g} {y_unit}/{x_unit}.")
        else:
            st.warning("The function is not real-valued at the current simulation point; no point or tangent is shown.")
    st.markdown("---")
    st.caption("Numerical root classification depends on solver behavior and tolerances. Rational/logarithmic functions may have domain restrictions or discontinuities; curves are split at non-finite values. A derivative describes the chosen mathematical model, not an observed real-world statistic.")
