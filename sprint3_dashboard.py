from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from sprint3_analytics import build_analytical_model, compare_customer_and_state_means, load_source_tables


ROOT = Path(__file__).resolve().parent
CSV_OUTPUT = ROOT / "DataDriven_TeamName_Sprint1.csv"
WORKBOOK = ROOT / "Sprint1_Cleaned_Pack.xlsx"
ORANGE = "#ff4b1f"
BLUE = "#168aad"
GREEN = "#16825d"
AMBER = "#d58a00"
RED = "#c83d32"
MUTED = "#64748b"


@st.cache_data(show_spinner="Loading cleaned BlueCart tables...")
def get_analytical_model() -> dict[str, Any]:
    return build_analytical_model(load_source_tables())


@st.cache_data(show_spinner="Validating the Sprint 2 baseline models...")
def evaluate_sprint2_models(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = orders.dropna(subset=["Total_Sales"]).copy()
    model_data = model_data.sort_values(
        ["Customer_ID_Masked", "Order_Date", "Order_ID"], na_position="last"
    )
    model_data["Prior_Purchases"] = model_data.groupby("Customer_ID_Masked")["Order_ID"].cumcount()
    model_data["Model_Month"] = model_data["Order_Date"].dt.month
    model_data["Model_Day_Of_Week"] = model_data["Order_Date"].dt.dayofweek

    feature_columns = ["State", "Prior_Purchases", "Model_Month", "Model_Day_Of_Week"]
    X = model_data[feature_columns]
    y = model_data["Total_Sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "state",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["State"],
            ),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                ["Prior_Purchases", "Model_Month", "Model_Day_Of_Week"],
            ),
        ],
        remainder="drop",
    )
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    rows = []
    linear_predictions = pd.DataFrame()
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        train_prediction = pipeline.predict(X_train)
        prediction = pipeline.predict(X_test)
        rows.append(
            {
                "Model": name,
            "Train MAE": mean_absolute_error(y_train, train_prediction),
            "Train RMSE": mean_squared_error(y_train, train_prediction) ** 0.5,
            "Train R²": r2_score(y_train, train_prediction),
                "Test MAE": mean_absolute_error(y_test, prediction),
                "Test RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                "Test R²": r2_score(y_test, prediction),
            }
        )
        if name == "Linear Regression":
            linear_predictions = pd.DataFrame(
                {"Actual": y_test.to_numpy(), "Predicted": prediction}
            )
    return pd.DataFrame(rows).sort_values("Test RMSE"), linear_predictions


def apply_sprint3_style() -> None:
    st.markdown(
        """
        <style>
                :root { --bluecart-ink:#0b1645; --bluecart-orange:#ff4b1f; --bluecart-line:#dfe5ee; }
                .stApp { background:#f5f7fb; color:#172033; }
                .block-container { max-width:1680px; padding: .55rem 1.25rem 2rem; }
                h1,h2,h3 { color:var(--bluecart-ink); letter-spacing:0; }
                h1 { font-size:1.95rem; line-height:1.15; margin:.15rem 0 .2rem; }
                h2 { font-size:1.25rem; }
                h3 { font-size:1.03rem; }
                [data-testid="stCaptionContainer"] { color:#68758a; }
                [data-testid="stMetric"] { min-height:92px; background:#fff; border:1px solid var(--bluecart-line); border-radius:10px; padding:.68rem .78rem; box-shadow:0 2px 7px rgba(11,22,69,.035); }
                [data-testid="stMetricLabel"] { font-size:.82rem; font-weight:650; color:#34415b; }
                [data-testid="stMetricValue"] { font-size:1.58rem; color:var(--bluecart-ink); }
                [data-testid="stMetricDelta"] { font-size:.72rem; }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) [data-testid="stMetric"] { background:#edf6ff; border-color:#cce4fb; }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stMetric"] { background:#edf9f2; border-color:#d0eedb; }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) [data-testid="stMetric"] { background:#fff9e9; border-color:#f4e7bf; }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4) [data-testid="stMetric"] { background:#f4f1ff; border-color:#e3dcfb; }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(5) [data-testid="stMetric"] { background:#fff0f2; border-color:#f3d8df; }
                section[data-testid="stSidebar"] { background:linear-gradient(180deg,#071333 0%,#0b1645 74%,#10245c 100%); border-right:1px solid #233766; }
                section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:.75rem .75rem 1rem; }
                section[data-testid="stSidebar"] * { color:#f6f8fc; }
                section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size:1.05rem; margin:.45rem 0 .65rem; }
                section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { font-size:.83rem; margin:.55rem 0 .25rem; color:#eaf0ff; }
                section[data-testid="stSidebar"] hr { border-color:#273963; margin:.75rem 0; }
                section[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.32rem .45rem; border-radius:7px; }
                section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#172951; color:#ff765b; }
                section[data-testid="stSidebar"] [data-testid="stSelectbox"] > label, section[data-testid="stSidebar"] [data-testid="stDateInput"] > label { font-size:.77rem; font-weight:650; color:#dfe7f8; }
                section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] { border-color:#56698f; background:#13234a; }
                div[data-testid="stExpander"] { background:#fff; border-color:var(--bluecart-line); border-radius:9px; }
                div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--bluecart-line); border-radius:10px; }
                .bc-question-band { background:#fff4ef; border:1px solid #ffded1; border-radius:10px; padding:.62rem .86rem; margin:.45rem 0 .75rem; color:#172033; font-size:.88rem; }
                .bc-question-band strong { color:#e8411a; }
                .bc-key-findings { background:#edf9f1; border:1px solid #caecd4; border-radius:10px; padding:.85rem 1rem; color:#193b2b; height:100%; }
                .bc-key-findings h3 { color:#155d3c; margin:0 0 .55rem; }
                .bc-finding { display:flex; gap:.55rem; align-items:flex-start; margin:.45rem 0; font-size:.82rem; line-height:1.35; }
                .bc-finding-num { display:inline-flex; flex:0 0 22px; width:22px; height:22px; align-items:center; justify-content:center; border-radius:50%; background:#d8f1e1; color:#155d3c; font-weight:800; }
                @media (max-width:900px) {
                    .block-container { padding:.65rem .8rem 1.5rem; }
                    h1 { font-size:1.55rem; }
                    [data-testid="stMetricValue"] { font-size:1.3rem; }
                    [data-testid="stMetric"] { min-height:78px; padding:.55rem; }
                }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _currency(value: float) -> str:
    return f"${value:,.2f}"


def _prepare_views(
    model: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    orders = model["orders"].copy()
    customers = model["customer_features"].copy()
    marketing = model["marketing"].copy()
    inventory = model["inventory"].copy()
    quality = model["quality"]

    page_names = [
        "Executive Overview",
        "Customer Data Mining",
        "Revenue & Trend Mining",
        "Marketing & Inventory",
        "Decision Support",
        "Ethics, Governance & Responsible Analytics",
    ]
    with st.sidebar:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:.55rem;padding:.1rem .2rem .7rem;font-size:1.16rem;font-weight:800'><span style='font-size:1.4rem'>🛒</span><span>BlueCart <span style='font-weight:500'>Retail</span></span></div>",
            unsafe_allow_html=True,
        )
        page = st.radio("Dashboard page", page_names, label_visibility="collapsed")
        st.divider()
        st.markdown("### Sales & customer filters")
        min_date = quality["date_min"]
        max_date = quality["date_max"]
        date_mode = st.radio("Order date", ["All available", "Select range"], horizontal=True)
        selected_dates = None
        if date_mode == "Select range" and pd.notna(min_date) and pd.notna(max_date):
            selected_dates = st.date_input(
                "Order date range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
        state_options = ["All"] + sorted(orders["State"].dropna().unique().tolist())
        selected_state = st.selectbox("State", state_options)
        selected_type = st.selectbox("Customer type", ["All", "Repeat", "One-Time"])
        segment_options = ["All"] + sorted(customers["Customer_Segment"].dropna().unique().tolist())
        selected_segment = st.selectbox("Customer segment", segment_options)
        st.divider()
        st.markdown("### Marketing filter")
        channels = sorted(marketing["Channel"].dropna().unique().tolist())
        selected_channel = st.selectbox("Marketing channel", ["All"] + channels)
        st.divider()
        if CSV_OUTPUT.exists():
            st.download_button(
                "Download cleaned CSV",
                data=CSV_OUTPUT.read_bytes(),
                file_name=CSV_OUTPUT.name,
                mime="text/csv",
                width="stretch",
            )
        if WORKBOOK.exists():
            st.download_button(
                "Download cleaned workbook",
                data=WORKBOOK.read_bytes(),
                file_name=WORKBOOK.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    scoped_orders = orders.copy()
    if selected_state != "All":
        scoped_orders = scoped_orders[scoped_orders["State"] == selected_state]
    if selected_dates and len(selected_dates) == 2:
        start_date, end_date = map(pd.Timestamp, selected_dates)
        scoped_orders = scoped_orders[
            scoped_orders["Order_Date"].between(start_date, end_date, inclusive="both")
        ]

    from sprint3_analytics import build_customer_features

    scoped_customers, thresholds = build_customer_features(scoped_orders)
    scoped_comparisons = compare_customer_and_state_means(scoped_orders)
    selected_customers = scoped_customers.copy()
    if selected_type != "All":
        selected_customers = selected_customers[
            selected_customers["Customer_Type"] == selected_type
        ]
    if selected_segment != "All":
        selected_customers = selected_customers[
            selected_customers["Customer_Segment"] == selected_segment
        ]
    if selected_type != "All" or selected_segment != "All":
        scoped_orders = scoped_orders[
            scoped_orders["Customer_ID_Masked"].isin(selected_customers["Customer_ID_Masked"])
        ]

    all_channels_marketing = marketing.copy()
    if selected_dates and len(selected_dates) == 2:
        all_channels_marketing = all_channels_marketing[
            all_channels_marketing["Campaign_Date"].between(start_date, end_date, inclusive="both")
        ]
    scoped_marketing = all_channels_marketing.copy()
    if selected_channel != "All":
        scoped_marketing = scoped_marketing[scoped_marketing["Channel"] == selected_channel]

    filters = {
        "page": page,
        "customers": selected_customers,
        "thresholds": thresholds,
        "state": selected_state,
        "type": selected_type,
        "segment": selected_segment,
        "channel": selected_channel,
        "date_mode": date_mode,
        "all_channels_marketing": all_channels_marketing,
        "statistical_comparisons": scoped_comparisons,
    }
    return scoped_orders, selected_customers, scoped_marketing, {**filters, "inventory": inventory, "quality": quality}


def _monthly_summary(orders: pd.DataFrame) -> pd.DataFrame:
    dated = orders.dropna(subset=["Order_Date"])
    if dated.empty:
        return pd.DataFrame(columns=["Month", "Revenue", "Transactions", "Observed_Sales_Transactions", "AOV"])
    dated = dated.assign(Month_Period=dated["Order_Date"].dt.to_period("M"))
    result = dated.groupby("Month_Period").agg(
        Revenue=("Total_Sales", lambda values: values.sum(min_count=1)),
        Transactions=("Order_ID", "nunique"),
        AOV=("Total_Sales", "mean"),
    )
    observed_counts = (
        dated.dropna(subset=["Total_Sales"])
        .groupby("Month_Period")["Order_ID"]
        .nunique()
        .rename("Observed_Sales_Transactions")
    )
    result = result.join(observed_counts).reset_index(names="Month")
    result["Observed_Sales_Transactions"] = result["Observed_Sales_Transactions"].fillna(0).astype(int)
    result["Month"] = result["Month"].astype(str)
    return result


def _state_summary(orders: pd.DataFrame) -> pd.DataFrame:
    observed = orders.dropna(subset=["Total_Sales"])
    if observed.empty:
        return pd.DataFrame(columns=["State", "Revenue", "Transactions", "Contribution"])
    result = (
        observed.groupby("State", as_index=False)
        .agg(Revenue=("Total_Sales", "sum"), Transactions=("Order_ID", "nunique"))
        .sort_values("Revenue", ascending=False)
    )
    total = result["Revenue"].sum()
    result["Contribution"] = result["Revenue"] / total if total else 0.0
    return result


def _segment_summary(customers: pd.DataFrame) -> pd.DataFrame:
    if customers.empty:
        return pd.DataFrame(
            columns=["Customer_Segment", "Customers", "Revenue", "Avg Customer AOV", "Avg Frequency"]
        )
    return (
        customers.groupby("Customer_Segment", as_index=False)
        .agg(
            Customers=("Customer_ID_Masked", "nunique"),
            Revenue=("Customer_Total_Revenue", lambda values: values.sum(min_count=1)),
            **{
                "Avg Customer AOV": ("Customer_Average_Order_Value", "mean"),
                "Avg Frequency": ("Customer_Frequency", "mean"),
            },
        )
        .sort_values("Revenue", ascending=False, na_position="last")
    )


def _chart(fig: Any, height: int = 350, percent_axis: str | None = None) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=16, r=16, t=50, b=16),
        font=dict(family="Arial, sans-serif", color="#172033"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9e0e8")
    fig.update_yaxes(gridcolor="#e9edf2", zerolinecolor="#d9e0e8")
    if percent_axis:
        fig.update_yaxes(tickformat=".1%", title=percent_axis)
    st.plotly_chart(fig, width="stretch")


def _metric_row(orders: pd.DataFrame, customers: pd.DataFrame, count: int = 5) -> dict[str, Any]:
    observed_sales = orders["Total_Sales"].dropna()
    revenue = float(observed_sales.sum()) if not observed_sales.empty else None
    aov = float(observed_sales.mean()) if not observed_sales.empty else None
    transaction_count = int(orders["Order_ID"].nunique(dropna=True))
    observed_customers = customers[customers["Customer_Purchase_Count"] > 0] if not customers.empty else customers
    customer_count = int(observed_customers["Customer_ID_Masked"].nunique()) if not observed_customers.empty else 0
    repeat_count = int(customers["Customer_Type"].eq("Repeat").sum()) if not customers.empty else 0
    repeat_rate = repeat_count / customer_count if customer_count else 0.0
    return {
        "revenue": revenue,
        "aov": aov,
        "transactions": transaction_count,
        "customers": customer_count,
        "repeat_count": repeat_count,
        "repeat_rate": repeat_rate,
    }


def _header(page: str) -> None:
    subtitles = {
        "BlueCart Retail — Data Mining & Decision Support": "Executive priorities from fragmented transactions, customer, campaign, and inventory records.",
        "Customer Behaviour & Segmentation": "Understand customer groups, purchase patterns, and value segments to identify retention opportunities and prioritise marketing actions.",
        "Revenue Patterns & Trends": "Compare monthly sales and state contribution; interpret trends in light of date completeness.",
        "Marketing & Inventory Intelligence": "Compare campaign engagement and stock indicators without implying sales attribution.",
        "BlueCart Decision Support & Alerts": "Transparent thresholds, actions, assumptions, risks, and governance for management review.",
    }
    heading, date_column = st.columns([0.84, 0.16], vertical_alignment="center")
    with heading:
        st.markdown(
            "<div style='color:#ff4b1f;font-weight:800;font-size:.75rem'>BLUECART RETAIL | FEDUNI INNOVATORS | SPRINT 3</div>",
            unsafe_allow_html=True,
        )
        st.title(page)
    with date_column:
        st.markdown("<div style='text-align:right;color:#64748b;font-size:.8rem;padding-top:1.1rem'>▦ 5 October 2026</div>", unsafe_allow_html=True)
    filter_note = st.session_state.get("sprint3_date_caption", "")
    subtitle = subtitles.get(page, "BlueCart Retail | Sprint 3")
    if filter_note:
        subtitle = f"{subtitle} {filter_note}"
    st.caption(subtitle)


def _show_overview(orders: pd.DataFrame, customers: pd.DataFrame, context: dict[str, Any]) -> None:
    _header("BlueCart Retail — Data Mining & Decision Support")
    st.markdown(
        "<div class='bc-question-band'><strong>Executive objective:</strong> Turn fragmented transaction, customer, campaign, and stock records into transparent priorities for revenue review, retention tests, and replenishment investigation.<br><strong>Business question:</strong> Where should BlueCart focus revenue review, customer retention, campaign testing, and replenishment attention?</div>",
        unsafe_allow_html=True,
    )
    kpis = _metric_row(orders, customers)
    cols = st.columns(5)
    cols[0].metric("Total revenue", _currency(kpis["revenue"]) if kpis["revenue"] is not None else "N/A", help="Observed sales only; transactions with missing sales are excluded.")
    cols[1].metric("Transactions", f"{kpis['transactions']:,}", help="Includes order IDs with missing sales values.")
    cols[2].metric("Unique customers", f"{kpis['customers']:,}")
    cols[3].metric("Average order value", _currency(kpis["aov"]) if kpis["aov"] is not None else "N/A")
    cols[4].metric("Repeat customer rate", f"{kpis['repeat_rate']:.1%}", help="Repeat customers divided by identified customers in the current scope.")

    inventory = context["inventory"]
    stock_values = inventory["Stock"]
    risk_cols = st.columns(4)
    risk_cols[0].metric("Inventory at/below 10", f"{int(stock_values.le(10).sum()):,}", help=f"{int(stock_values.eq(0).sum())} critical zero-stock, {int((stock_values.gt(0) & stock_values.le(10)).sum())} positive-stock warnings, and {int(stock_values.isna().sum())} missing stock values.")
    risk_cols[1].metric("Campaigns", f"{int(context['marketing']['Campaign_ID'].nunique()):,}", help="Campaign records; no reliable campaign-to-order linkage.")
    model_results = context["model_results"]
    linear_result = model_results.loc[model_results["Model"] == "Linear Regression"].iloc[0]
    risk_cols[2].metric("Baseline model R²", f"{linear_result['Test R²']:.3f}", help="Negative R² indicates weak out-of-sample performance; not production-ready.")
    risk_cols[3].metric("Model status", "Weak baseline", help="Use descriptive patterns and human-reviewed rules, not automated sales forecasts.")

    state = _state_summary(orders)
    named_states = state[state["State"] != "Unassigned"]
    segments = _segment_summary(customers)
    marketing = context["marketing"]
    channels = marketing.dropna(subset=["CTR"]).groupby("Channel")["CTR"].mean().sort_values(ascending=False)
    zero_stock = int(inventory["Stock"].eq(0).sum())
    comparisons = context["statistical_comparisons"]
    state_test = comparisons["state_anova"]
    customer_test = comparisons["customer_welch"]
    left, middle, right = st.columns([1.18, 1.18, 1.0], gap="small")
    monthly = _monthly_summary(orders)
    with left:
        st.subheader("▮  Monthly revenue")
        if not monthly.empty:
            revenue_chart = px.bar(
                monthly,
                x="Month",
                y="Revenue",
                color="Month",
                color_discrete_map={"2025-01": "#1485df", "2025-02": "#1485df", "2025-03": "#0d62ac"},
                text=monthly["Revenue"].map(lambda value: f"${value:,.0f}" if pd.notna(value) else "N/A"),
                hover_data={"Observed_Sales_Transactions": True, "Transactions": True, "AOV": ":$,.2f"},
                title=None,
                labels={"Month": "Month", "Revenue": "Revenue"},
            )
            revenue_chart.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
            revenue_chart.update_layout(height=255, margin=dict(l=8, r=10, t=18, b=8), bargap=0.34)
            revenue_chart.update_xaxes(type="category", ticktext=[pd.Period(x).strftime("%b %Y") for x in monthly["Month"]], tickvals=monthly["Month"], title=None)
            revenue_chart.update_yaxes(title="Revenue", tickprefix="$", separatethousands=True, rangemode="tozero")
            _chart(revenue_chart, height=260)
            march_count = int(monthly.loc[monthly["Month"].eq("2025-03"), "Observed_Sales_Transactions"].iloc[0]) if monthly["Month"].eq("2025-03").any() else 0
            st.caption(f"⚠ Sparse March: only {march_count} dated sales. {context['quality']['order_dates_missing']} order dates are missing.")
        else:
            st.info("No dated orders in this view.")
    with middle:
        st.subheader("●  Revenue by state")
        if not state.empty:
            state_chart = px.bar(
                state,
                x="Revenue",
                y="State",
                orientation="h",
                text=state["Revenue"].map(lambda value: f"${value:,.0f}"),
                color="State",
                color_discrete_map={"Washington": ORANGE, "Oregon": BLUE, "California": BLUE, "Unassigned": "#d58a00"},
                title=None,
                hover_data={"Revenue": ":$,.2f", "Contribution": ":.1%", "Transactions": True},
            )
            state_chart.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
            state_chart.update_layout(height=260, margin=dict(l=6, r=20, t=18, b=8))
            state_chart.update_xaxes(title="Revenue", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="#edf1f6")
            state_chart.update_yaxes(title=None, categoryorder="total ascending")
            _chart(state_chart, height=260)
            st.caption(f"ANOVA p={state_test['p_value']:.3f}: state mean transaction values are not significantly different.")
        else:
            st.info("No observed revenue in this view.")
    with right:
        st.subheader("♟  Customer behaviour")
        behaviour = pd.DataFrame(
            [
                {"Type": "Repeat customers", "Orders": customer_test["repeat_n"], "Mean value": _currency(customer_test["repeat_mean"])},
                {"Type": "One-time", "Orders": customer_test["one_time_n"], "Mean value": _currency(customer_test["one_time_mean"])},
            ]
        )
        st.dataframe(behaviour, width="stretch", hide_index=True, height=146)
        st.caption(f"Welch p={customer_test['p_value']:.3f}: mean-value difference is not statistically significant.")

    left, middle, right = st.columns([1.08, 1.08, 1.0], gap="small")
    with left:
        st.subheader("▤  Customer segment distribution")
        if not segments.empty:
            count_chart = px.bar(
                segments.sort_values("Customers"),
                x="Customers",
                y="Customer_Segment",
                orientation="h",
                text="Customers",
                color_discrete_sequence=[BLUE],
                title=None,
            )
            count_chart.update_traces(textposition="outside", cliponaxis=False)
            count_chart.update_layout(height=280, margin=dict(l=4, r=20, t=12, b=8))
            count_chart.update_xaxes(title="Customers", showgrid=True, gridcolor="#edf1f6")
            count_chart.update_yaxes(title=None, tickfont=dict(size=10))
            _chart(count_chart, height=280)
        else:
            st.info("No customer segments are available in this view.")
    with middle:
        st.subheader("🏆  Top segments by revenue")
        revenue_segments = segments.dropna(subset=["Revenue"]).nlargest(6, "Revenue")
        if not revenue_segments.empty:
            segment_chart = px.bar(
                revenue_segments.sort_values("Revenue"),
                x="Revenue",
                y="Customer_Segment",
                orientation="h",
                text=revenue_segments.sort_values("Revenue")["Revenue"].map(lambda value: f"${value:,.0f}"),
                title=None,
                color_discrete_sequence=[ORANGE],
                hover_data={"Customers": True, "Revenue": ":$,.2f"},
            )
            segment_chart.update_traces(textposition="outside", cliponaxis=False)
            segment_chart.update_layout(height=280, margin=dict(l=6, r=20, t=12, b=8))
            segment_chart.update_xaxes(title="Observed revenue", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="#edf1f6")
            segment_chart.update_yaxes(title=None, tickfont=dict(size=10))
            _chart(segment_chart, height=280)
        else:
            st.info("No observed customer revenue in this view.")
    with right:
        state_lead = named_states.iloc[0] if not named_states.empty else None
        lead_segment = segments.dropna(subset=["Revenue"]).iloc[0] if segments["Revenue"].notna().any() else None
        if state_lead is None:
            insight_state = "No named state leads in the current filtered selection."
        else:
            insight_state = f"{state_lead['State']} leads named markets at {_currency(float(state_lead['Revenue']))} ({state_lead['Contribution']:.1%} of observed revenue)."
        insight_customer = (
            f"Repeat customers account for {customer_test['repeat_n']} observed orders vs {customer_test['one_time_n']} one-time; mean values are not significantly different."
            if customer_test["p_value"] is not None
            else "Customer-type comparison is unavailable in this filtered sample."
        )
        insight_value = (
            f"{lead_segment['Customer_Segment']} is the largest customer revenue segment ({_currency(float(lead_segment['Revenue']))})."
            if lead_segment is not None
            else "No customer revenue segment is available in this selection."
        )
        insight_channel = (
            f"{channels.index[0]} has the highest average CTR ({channels.iloc[0]:.2%}); this measures engagement, not ROI."
            if len(channels)
            else "Campaign CTR is unavailable in this selection."
        )
        insight_inventory = f"{zero_stock} products have recorded zero stock; review availability. Inventory is not linked to orders by SKU."
        insight_items = [insight_state, insight_customer, insight_value, insight_channel, insight_inventory]
        insight_html = "<div class='bc-key-findings'><h3>💡 Key findings</h3>"
        insight_html += "".join(
            f"<div class='bc-finding'><span class='bc-finding-num'>{idx}</span><span>{item}</span></div>"
            for idx, item in enumerate(insight_items, start=1)
        )
        insight_html += "</div>"
        st.markdown(insight_html, unsafe_allow_html=True)
    st.caption("Source note: sales exclude 75 nonpositive source transactions; 16 identified customer IDs have no valid sales. Marketing and inventory remain separate from transaction records.")


def _show_customer_mining(orders: pd.DataFrame, customers: pd.DataFrame, thresholds: dict[str, Any], comparisons: dict[str, Any]) -> None:
    _header("Customer Behaviour & Segmentation")
    metrics = _metric_row(orders, customers)
    valid_customers = customers[customers["Customer_Purchase_Count"] > 0].copy() if not customers.empty else customers
    repeat = int(valid_customers["Customer_Type"].eq("Repeat").sum()) if not valid_customers.empty else 0
    one_time = int(valid_customers["Customer_Type"].eq("One-Time").sum()) if not valid_customers.empty else 0
    mean_customer_revenue = valid_customers["Customer_Total_Revenue"].mean() if not valid_customers.empty else np.nan
    high_value_mask = customers["Customer_Total_Revenue"].gt(thresholds["revenue_p75"]) if not customers.empty else pd.Series(dtype=bool)
    high_value = int(high_value_mask.sum()) if not customers.empty else 0
    repeat_rate = repeat / metrics["customers"] if metrics["customers"] else 0.0
    one_time_rate = one_time / metrics["customers"] if metrics["customers"] else 0.0
    cols = st.columns(5)
    cols[0].metric("Unique customers", f"{metrics['customers']:,}", help="Customers with at least one observed positive-sale transaction.")
    cols[1].metric("Repeat customers", f"{repeat:,}", f"{repeat_rate:.1%} of customers")
    cols[2].metric("One-time customers", f"{one_time:,}", f"{one_time_rate:.1%} of customers")
    cols[3].metric("Average customer revenue", _currency(float(mean_customer_revenue)) if pd.notna(mean_customer_revenue) else "N/A", help="Mean of observed customer totals; customers without valid sales are excluded.")
    cols[4].metric("High-value customers", f"{high_value:,}", help=f"Observed customer revenue above P75 ({_currency(thresholds['revenue_p75'])}).")

    summary = _segment_summary(customers) if not customers.empty else pd.DataFrame()
    customer_test = comparisons["customer_welch"]
    repeat_revenue = float(valid_customers.loc[valid_customers["Customer_Type"].eq("Repeat"), "Customer_Total_Revenue"].sum()) if not valid_customers.empty else 0.0
    identified_revenue = float(valid_customers["Customer_Total_Revenue"].sum()) if not valid_customers.empty else 0.0
    repeat_revenue_share = repeat_revenue / identified_revenue if identified_revenue else 0.0
    high_value_revenue = float(customers.loc[high_value_mask, "Customer_Total_Revenue"].sum()) if not customers.empty else 0.0
    high_value_share = high_value_revenue / identified_revenue if identified_revenue else 0.0
    missing_sales_customers = int(customers["Customer_Type"].eq("Insufficient Sales Data").sum()) if not customers.empty else 0
    dated_orders = orders.dropna(subset=["Order_Date"])
    reference_date = dated_orders["Order_Date"].max() if not dated_orders.empty else None
    observed_recency_customers = int(customers["Customer_Recency"].notna().sum()) if not customers.empty else 0
    recency_days_missing = int(orders["Order_Date"].isna().sum())

    top, mid, insights = st.columns([1.08, 1.08, 0.94], gap="small")
    with top:
        st.subheader("👥  Customer count by segment")
        if not summary.empty:
            count_chart = px.bar(
                summary.sort_values("Customers"),
                x="Customers",
                y="Customer_Segment",
                orientation="h",
                text="Customers",
                title=None,
                color_discrete_sequence=[BLUE],
            )
            count_chart.update_traces(textposition="outside", cliponaxis=False)
            count_chart.update_layout(height=235, margin=dict(l=8, r=18, t=8, b=8), bargap=.24)
            count_chart.update_xaxes(title="Customers", showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
            count_chart.update_yaxes(title=None, tickfont=dict(size=10))
            _chart(count_chart, height=235)
            st.caption("One-time/repeat use purchase counts; High Value means observed customer revenue > P75.")
        else:
            st.info("No customer segments available for this selection.")
    with mid:
        st.subheader("💲  Observed revenue by segment")
        revenue_summary = summary.dropna(subset=["Revenue"]).sort_values("Revenue") if not summary.empty else pd.DataFrame()
        if not revenue_summary.empty:
            segment_revenue_chart = px.bar(
                revenue_summary,
                x="Revenue",
                y="Customer_Segment",
                orientation="h",
                text=revenue_summary["Revenue"].map(lambda value: f"${value:,.0f}"),
                title=None,
                color_discrete_sequence=[GREEN],
                hover_data={"Customers": True, "Revenue": ":$,.2f", "Avg Customer AOV": ":$,.2f", "Avg Frequency": ":.2f"},
            )
            segment_revenue_chart.update_traces(textposition="outside", cliponaxis=False)
            segment_revenue_chart.update_layout(height=235, margin=dict(l=8, r=18, t=8, b=8), bargap=.24)
            segment_revenue_chart.update_xaxes(title="Observed revenue", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
            segment_revenue_chart.update_yaxes(title=None, tickfont=dict(size=10))
            _chart(segment_revenue_chart, height=235)
            st.caption(f"High value: customer revenue > P75 ({_currency(thresholds['revenue_p75'])}); {high_value} customers qualify.")
        else:
            st.info("No observed customer revenue available for this selection.")
    with insights:
        items = [
            f"High-value customers (n={high_value}) generate {high_value_share:.1%} of identified customer revenue ({_currency(high_value_revenue)}).",
            (
                f"Repeat customers (n={repeat}) account for {repeat_revenue_share:.1%} of identified customer revenue; transaction-value difference is not significant (p={customer_test['p_value']:.3f})."
                if customer_test["p_value"] is not None
                else "Repeat/one-time comparison is unavailable for this filtered sample."
            ),
            f"One-time customers (n={one_time}) represent a retention opportunity; test treatment impact rather than assuming an uplift.",
            f"{missing_sales_customers} identified customers have insufficient valid sales for monetary segmentation.",
            "No churn or at-risk status is assigned because order dates are incomplete and the observation window is short.",
        ]
        insight_markup = "<div class='bc-key-findings'><h3>💡 Key insights</h3>"
        insight_markup += "".join(
            f"<div class='bc-finding'><span class='bc-finding-num'>{index}</span><span>{item}</span></div>"
            for index, item in enumerate(items, start=1)
        )
        insight_markup += "</div>"
        st.markdown(insight_markup, unsafe_allow_html=True)

    frequency, recency = st.columns(2, gap="small")
    with frequency:
        st.subheader("📈  Purchase frequency versus customer revenue")
        revenue_customers = valid_customers.dropna(subset=["Customer_Total_Revenue"]) if not valid_customers.empty else valid_customers
        if not revenue_customers.empty:
            frequency_chart = px.scatter(
                revenue_customers,
                x="Customer_Frequency",
                y="Customer_Total_Revenue",
                color="Customer_Segment",
                hover_data=["Customer_ID_Masked", "Customer_Type", "Customer_Recency", "RFM_Score"],
                title=None,
                labels={"Customer_Frequency": "Observed purchase count", "Customer_Total_Revenue": "Customer revenue"},
                color_discrete_sequence=[BLUE, ORANGE, GREEN, AMBER, RED],
            )
            frequency_chart.update_traces(marker=dict(size=8, opacity=.8))
            frequency_chart.update_layout(height=245, margin=dict(l=8, r=12, t=8, b=8), legend=dict(font=dict(size=9), orientation="h", y=1.12, x=0))
            frequency_chart.update_xaxes(title="Observed purchase count", dtick=1, showgrid=True, gridcolor="#edf1f6")
            frequency_chart.update_yaxes(title="Customer revenue", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="#edf1f6")
            _chart(frequency_chart, height=245)
        else:
            st.info("No customer revenue points available in this filtered view.")
    with recency:
        st.subheader("◷  Customer recency")
        recency_data = customers.dropna(subset=["Customer_Recency"]) if not customers.empty else customers
        if not recency_data.empty:
            recency_chart = px.histogram(
                recency_data,
                x="Customer_Recency",
                nbins=12,
                title=None,
                color_discrete_sequence=["#1189ae"],
                labels={"Customer_Recency": "Days since latest dated purchase"},
            )
            recency_chart.update_layout(height=245, margin=dict(l=8, r=12, t=8, b=8), bargap=.08)
            recency_chart.update_xaxes(title="Days since latest dated purchase", showgrid=True, gridcolor="#edf1f6")
            recency_chart.update_yaxes(title="Customers", showgrid=True, gridcolor="#edf1f6")
            _chart(recency_chart, height=245)
        else:
            st.info("Recency is unavailable because no dated purchases remain in this view.")

    table_left, table_right = st.columns([1.05, .95], gap="small")
    with table_left:
        st.subheader("🏆  Top 10 customers by observed revenue")
        top_customers = valid_customers.dropna(subset=["Customer_Total_Revenue"]).nlargest(10, "Customer_Total_Revenue") if not valid_customers.empty else valid_customers
        if not top_customers.empty:
            top_table = top_customers[["Customer_ID_Masked", "Customer_Segment", "Customer_Total_Revenue", "Customer_Purchase_Count", "Customer_Average_Order_Value", "Customer_Recency"]].rename(
                columns={
                    "Customer_ID_Masked": "Customer ID",
                    "Customer_Segment": "Segment",
                    "Customer_Total_Revenue": "Revenue",
                    "Customer_Purchase_Count": "Orders",
                    "Customer_Average_Order_Value": "Average order value",
                    "Customer_Recency": "Recency (days)",
                }
            ).copy()
            for column in ["Revenue", "Average order value"]:
                top_table[column] = top_table[column].map(lambda value: _currency(value) if pd.notna(value) else "N/A")
            top_table["Recency (days)"] = top_table["Recency (days)"].map(lambda value: f"{int(value)}" if pd.notna(value) else "N/A")
            st.dataframe(top_table, width="stretch", hide_index=True, height=286)
        else:
            st.info("No valid customer revenue in this selection.")
    with table_right:
        st.subheader("▤  Customer segment detail")
        if not summary.empty:
            detail = summary.rename(
                columns={
                    "Customer_Segment": "Segment",
                    "Avg Customer AOV": "Avg. customer value",
                    "Avg Frequency": "Avg. frequency",
                }
            ).copy()
            detail["Revenue"] = detail["Revenue"].map(lambda value: _currency(value) if pd.notna(value) else "N/A")
            detail["Avg. customer value"] = detail["Avg. customer value"].map(lambda value: _currency(value) if pd.notna(value) else "N/A")
            detail["Avg. frequency"] = detail["Avg. frequency"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "N/A")
            st.dataframe(detail, width="stretch", hide_index=True, height=286)
        else:
            st.info("No customer segments available.")
        st.download_button(
            "Download masked customer table",
            data=customers.to_csv(index=False).encode("utf-8"),
            file_name="BlueCart_Customer_Features.csv",
            mime="text/csv",
        )

    recency_note = (
        f"Recency is calculated relative to {reference_date:%d %b %Y} for {observed_recency_customers} customers with dated observed sales. "
        f"{recency_days_missing} transaction order dates are missing. Segments use observed sales and do not imply predicted churn."
        if reference_date is not None
        else f"Recency is unavailable in this filter. {recency_days_missing} transaction dates are missing; segments do not imply predicted churn."
    )
    st.markdown(
        f"<div style='background:#fff7e4;border:1px solid #f4df9d;border-radius:8px;padding:.52rem .75rem;color:#604a11;font-size:.8rem'><strong>⚠ Data note:</strong> {recency_note}</div>",
        unsafe_allow_html=True,
    )


def _show_revenue_trends(orders: pd.DataFrame, context: dict[str, Any]) -> None:
    _header("Revenue Patterns & Trends")
    monthly = _monthly_summary(orders)
    state = _state_summary(orders)
    known_states = state[state["State"] != "Unassigned"]
    observed_months = monthly.dropna(subset=["Revenue"])
    if not observed_months.empty:
        best_month = observed_months.loc[observed_months["Revenue"].idxmax()]
        lowest_month = observed_months.loc[observed_months["Revenue"].idxmin()]
    else:
        best_month = lowest_month = None
    march_rows = monthly.loc[monthly["Month"].eq("2025-03"), "Observed_Sales_Transactions"]
    march_observed = int(march_rows.iloc[0]) if not march_rows.empty else 0
    comparison = context.get("statistical_comparisons", {}).get("state_anova", {})
    state_test_p = comparison.get("p_value")
    median_state_revenue = float(known_states["Revenue"].median()) if not known_states.empty else 0.0
    if best_month is not None:
        highest_month_label = pd.Period(str(best_month["Month"]), freq="M").strftime("%b %Y")
        lowest_month_label = pd.Period(str(lowest_month["Month"]), freq="M").strftime("%b %Y")
        highest_month_revenue = f"USD {float(best_month['Revenue']):,.2f}"
        lowest_month_revenue = f"USD {float(lowest_month['Revenue']):,.2f}"
    else:
        highest_month_label = lowest_month_label = "N/A"
        highest_month_revenue = lowest_month_revenue = "N/A"

    summary_cols = st.columns(5)
    summary_cols[0].metric("Top named state", known_states.iloc[0]["State"] if not known_states.empty else "N/A")
    summary_cols[1].metric("Top state revenue", _currency(float(known_states.iloc[0]["Revenue"])) if not known_states.empty else "N/A")
    summary_cols[2].metric("Lowest named state", known_states.iloc[-1]["State"] if not known_states.empty else "N/A")
    summary_cols[3].metric("Highest observed month", highest_month_label)
    summary_cols[4].metric("Lowest observed month", lowest_month_label)

    if best_month is not None:
        st.info(
            f"Highest dated-month bucket: {highest_month_label} ({highest_month_revenue}, "
            f"{int(best_month['Observed_Sales_Transactions'])} dated sales). Lowest: {lowest_month_label} "
            f"({lowest_month_revenue}, {int(lowest_month['Observed_Sales_Transactions'])} dated sales). "
            f"{context['quality']['order_dates_missing']} transaction dates are missing; March has only {march_observed} dated sales and is sparse."
        )

    top_row_left, top_row_right = st.columns([1, 1], gap="small")
    with top_row_left:
        with st.container(border=True):
            st.subheader("▮  Monthly observed revenue")
            st.caption("Only dated transactions with observed sales")
            if not observed_months.empty:
                monthly_plot = observed_months.copy()
                monthly_plot["Month_Label"] = monthly_plot["Month"].map(
                    lambda value: pd.Period(value, freq="M").strftime("%b %Y")
                )
                monthly_plot["Point_Label"] = monthly_plot.apply(
                    lambda row: f"{_currency(float(row['Revenue']))}<br>n={int(row['Observed_Sales_Transactions'])}",
                    axis=1,
                )
                monthly_chart = px.line(
                    monthly_plot,
                    x="Month_Label",
                    y="Revenue",
                    markers=True,
                    text="Point_Label",
                    title=None,
                    custom_data=["Observed_Sales_Transactions", "Transactions", "AOV"],
                    labels={"Month_Label": "Month", "Revenue": "Revenue"},
                )
                monthly_chart.update_traces(
                    line=dict(color=BLUE, width=3),
                    marker=dict(size=10, color=[RED if label == "Mar 2025" else BLUE for label in monthly_plot["Month_Label"]]),
                    textposition="top center",
                    textfont=dict(size=11, color="#24314a"),
                    hovertemplate="%{x}<br>Revenue: $%{y:,.2f}<br>Dated sales: %{customdata[0]}<br>AOV: $%{customdata[2]:,.2f}<extra></extra>",
                )
                monthly_chart.update_layout(height=300, margin=dict(l=8, r=12, t=26, b=8), showlegend=False)
                monthly_chart.update_xaxes(type="category", categoryorder="array", categoryarray=monthly_plot["Month_Label"].tolist(), title=None, showgrid=False)
                monthly_chart.update_yaxes(title="Revenue", tickprefix="$", separatethousands=True, rangemode="tozero", showgrid=True, gridcolor="#edf1f6")
                _chart(monthly_chart, height=300)
                st.caption(f"March is sparse: {march_observed} dated sales with an observed value.")
            else:
                st.info("No observed sales values in dated orders for this selection.")

    with top_row_right:
        with st.container(border=True):
            st.subheader("📍  Revenue contribution by state")
            st.caption("Highest observed revenue; mean differences are not significant")
            if not state.empty:
                state_plot = state.sort_values("Revenue", ascending=True).copy()
                state_plot["Label"] = state_plot.apply(
                    lambda row: f"{_currency(float(row['Revenue']))}  (n={int(row['Transactions'])})",
                    axis=1,
                )
                state_chart = px.bar(
                    state_plot,
                    x="Revenue",
                    y="State",
                    orientation="h",
                    text="Label",
                    color="State",
                    color_discrete_map={"Washington": ORANGE, "Oregon": BLUE, "California": BLUE, "Unassigned": AMBER},
                    title=None,
                    hover_data={"Revenue": ":$,.2f", "Contribution": ":.1%", "Transactions": True, "Label": False},
                )
                state_chart.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
                state_chart.update_layout(height=300, margin=dict(l=8, r=24, t=26, b=8))
                state_chart.update_xaxes(title="Revenue", tickprefix="$", separatethousands=True, showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
                state_chart.update_yaxes(title=None, categoryorder="total ascending")
                _chart(state_chart, height=300)
                if state_test_p is not None:
                    st.markdown(
                        f"<div style='background:#edf4ff;border:1px solid #d5e5fa;border-radius:7px;padding:.45rem .65rem;font-size:.78rem;color:#203b63'><strong>ANOVA p={state_test_p:.3f}</strong> · No significant state mean transaction-value difference. Unassigned is excluded from named-state tests.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No observed revenue in this selection.")

    bottom_left, bottom_right = st.columns([1, 1], gap="small")
    with bottom_left:
        with st.container(border=True):
            st.subheader("📈  Average order value by dated month")
            st.caption("Mean observed transaction value; labels show the calendar month")
            aov_months = monthly.dropna(subset=["AOV"]).copy()
            if not aov_months.empty:
                aov_months["Month_Label"] = aov_months["Month"].map(
                    lambda value: pd.Period(value, freq="M").strftime("%b %Y")
                )
                aov_chart = px.line(
                    aov_months,
                    x="Month_Label",
                    y="AOV",
                    markers=True,
                    text=aov_months["AOV"].map(lambda value: f"${value:,.2f}"),
                    title=None,
                    labels={"Month_Label": "Month", "AOV": "Average order value"},
                )
                aov_chart.update_traces(line=dict(color=BLUE, width=3), marker=dict(size=9, color=BLUE), textposition="top center")
                aov_chart.update_layout(height=270, margin=dict(l=8, r=14, t=25, b=8), showlegend=False)
                aov_chart.update_xaxes(type="category", categoryorder="array", categoryarray=aov_months["Month_Label"].tolist(), title=None, showgrid=False)
                aov_chart.update_yaxes(title="Average order value", tickprefix="$", showgrid=True, gridcolor="#edf1f6", rangemode="normal")
                _chart(aov_chart, height=270)
            else:
                st.info("No monthly average is available for this selection.")

    with bottom_right:
        state_lead = known_states.iloc[0] if not known_states.empty else None
        highest_month_text = (
            f"{highest_month_label} has the highest dated revenue (USD {float(best_month['Revenue']):,.2f})."
            if best_month is not None
            else "No dated monthly revenue is available in this selection."
        )
        state_text = (
            f"{state_lead['State']} leads named markets at USD {float(state_lead['Revenue']):,.2f} ({state_lead['Contribution']:.1%} of observed revenue)."
            if state_lead is not None
            else "No named-state revenue is available in this selection."
        )
        p_text = (
            f"The state mean transaction comparison is not significant (ANOVA p={state_test_p:.3f}; n={comparison['n']})."
            if state_test_p is not None
            else "State comparison is unavailable for this small selection."
        )
        contribution_note = "Revenue includes Unassigned; this is not a named market and is excluded from named-state ranking."
        insight_items = [
            state_text,
            p_text,
            highest_month_text,
            f"March has {march_observed} dated sales; {context['quality']['order_dates_missing']} transaction dates are missing.",
            contribution_note,
        ]
        insight_markup = "<div class='bc-key-findings'><h3>💡 Key insights</h3>"
        insight_markup += "".join(
            f"<div class='bc-finding'><span class='bc-finding-num'>{index}</span><span>{item}</span></div>"
            for index, item in enumerate(insight_items, start=1)
        )
        insight_markup += "</div>"
        st.markdown(insight_markup, unsafe_allow_html=True)


def _show_marketing_inventory(marketing: pd.DataFrame, inventory: pd.DataFrame) -> None:
    _header("Marketing & Inventory Intelligence")
    campaigns = (
        marketing.groupby("Channel", as_index=False)
        .agg(
            Campaigns=("Campaign_ID", "nunique"),
            Average_CTR=("CTR", "mean"),
            Average_Reach=("Reach", "mean"),
            CTR_Observed=("CTR", "count"),
        )
        .sort_values("Average_CTR", ascending=False)
    )
    stock = inventory.dropna(subset=["Stock"]).copy()
    zero_stock = int(stock["Stock"].eq(0).sum())
    low_stock = int((stock["Stock"].gt(0) & stock["Stock"].le(10)).sum())
    missing_stock = int(inventory["Stock"].isna().sum())
    at_or_below_ten = int(inventory["Stock"].le(10).sum())
    avg_ctr = float(marketing["CTR"].mean()) if marketing["CTR"].notna().any() else None
    missing_ctr = int(marketing["CTR"].isna().sum())
    observed_ctr = marketing.dropna(subset=["CTR"]).copy()
    ctr_channel_median = float(campaigns["Average_CTR"].median()) if campaigns["Average_CTR"].notna().any() else None
    high_ctr_channels = campaigns.loc[campaigns["Average_CTR"] >= campaigns["Average_CTR"].quantile(.75), "Channel"].tolist() if campaigns["Average_CTR"].notna().any() else []
    low_ctr_channels = campaigns.loc[campaigns["Average_CTR"] < ctr_channel_median, "Channel"].tolist() if ctr_channel_median is not None else []
    stock_bands = inventory.copy()
    stock_bands["Stock band"] = np.select(
        [
            stock_bands["Stock"].isna(),
            stock_bands["Stock"].eq(0),
            stock_bands["Stock"].between(1, 10, inclusive="both"),
            stock_bands["Stock"].between(11, 25, inclusive="both"),
            stock_bands["Stock"].between(26, 50, inclusive="both"),
        ],
        ["Missing", "0", "1-10", "11-25", "26-50"],
        default="51+",
    )
    stock_band_order = ["Missing", "0", "1-10", "11-25", "26-50", "51+"]
    stock_band_counts = (
        stock_bands["Stock band"].value_counts()
        .reindex(stock_band_order, fill_value=0)
        .rename_axis("Stock band")
        .reset_index(name="SKUs")
    )
    stock_band_counts = stock_band_counts[stock_band_counts["SKUs"] > 0]

    cols = st.columns(5)
    cols[0].metric("Campaigns", f"{int(marketing['Campaign_ID'].nunique()):,}")
    cols[1].metric("Average CTR", f"{avg_ctr:.2%}" if avg_ctr is not None else "N/A")
    cols[2].metric("Total SKUs", f"{int(inventory['SKU'].nunique()):,}")
    cols[3].metric("Zero-stock products", f"{zero_stock:,}")
    cols[4].metric("Stock <= 10", f"{at_or_below_ten:,}", help=f"{zero_stock} critical at zero; {low_stock} positive-stock warnings; {missing_stock} missing value excluded from this total.")
    st.caption(f"Stock alerts: CRITICAL {zero_stock} at zero | WARNING {low_stock} positive units <=10 | DATA QUALITY {missing_stock} missing. Totals describe independent campaign and inventory tables.")

    campaign_left, campaign_right = st.columns([1, 1], gap="small")
    with campaign_left:
        with st.container(border=True):
            st.subheader("▮  Average CTR by channel")
            st.caption("CTR and reach are engagement measures, not sales attribution.")
            if not campaigns.empty and campaigns["Average_CTR"].notna().any():
                channel_plot = campaigns.dropna(subset=["Average_CTR"]).sort_values("Average_CTR").copy()
                channel_plot["CTR label"] = channel_plot["Average_CTR"].map(lambda value: f"{value:.2%}")
                channel_chart = px.bar(
                    channel_plot,
                    x="Average_CTR",
                    y="Channel",
                    orientation="h",
                    text="CTR label",
                    title=None,
                    color_discrete_sequence=[BLUE],
                    hover_data={"Campaigns": True, "Average_CTR": ":.2%", "Average_Reach": ":,.0f", "CTR_Observed": True, "CTR label": False},
                )
                channel_chart.update_traces(textposition="outside", cliponaxis=False)
                channel_chart.update_layout(height=285, margin=dict(l=8, r=25, t=12, b=8), bargap=.25)
                channel_chart.update_xaxes(title="Mean CTR", tickformat=".1%", showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
                channel_chart.update_yaxes(title=None, tickfont=dict(size=10))
                _chart(channel_chart, height=285)
            else:
                st.info("No observed campaign CTR for this selection.")
    with campaign_right:
        with st.container(border=True):
            st.subheader("⚠  Campaign CTR distribution")
            st.caption("Distribution of observed campaign CTR values")
            if not observed_ctr.empty:
                observed_ctr["CTR band"] = observed_ctr["CTR"].map(lambda value: f"{value:.1%}")
                ctr_counts = (
                    observed_ctr.groupby("CTR", as_index=False)
                    .agg(Campaigns=("Campaign_ID", "nunique"))
                    .sort_values("CTR")
                )
                ctr_counts["CTR label"] = ctr_counts["CTR"].map(lambda value: f"{value:.1%}")
                ctr_chart = px.bar(
                    ctr_counts,
                    x="CTR label",
                    y="Campaigns",
                    text="Campaigns",
                    title=None,
                    color_discrete_sequence=[ORANGE],
                    hover_data={"CTR": ":.2%", "Campaigns": True},
                )
                ctr_chart.update_traces(textposition="outside", cliponaxis=False)
                ctr_chart.update_layout(height=285, margin=dict(l=8, r=14, t=12, b=8), bargap=.24)
                ctr_chart.update_xaxes(title="CTR band", type="category", categoryorder="array", categoryarray=ctr_counts["CTR label"].tolist(), showgrid=False)
                ctr_chart.update_yaxes(title="Campaigns", showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
                _chart(ctr_chart, height=285)
            else:
                st.info("No CTR values to chart.")

    with st.container(border=True):
        st.subheader("▤  Channel performance detail")
        st.caption(f"Mean CTR is {avg_ctr:.2%} across available values; {missing_ctr} CTR values are missing. Reach and CTR are summarized independently; no campaign ROI is inferred.")
        if not campaigns.empty:
            display_campaigns = campaigns.copy()
            display_campaigns["Average_CTR"] = display_campaigns["Average_CTR"].map(lambda value: f"{value:.2%}" if pd.notna(value) else "N/A")
            display_campaigns["Average_Reach"] = display_campaigns["Average_Reach"].map(lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A")
            display_campaigns = display_campaigns.rename(columns={"Campaigns": "Campaign count", "Average_CTR": "Mean CTR", "Average_Reach": "Mean reach", "CTR_Observed": "CTR observations"})
            st.dataframe(display_campaigns, width="stretch", hide_index=True)
        else:
            st.info("No campaign rows match the current filters.")

    inventory_left, inventory_right = st.columns([1, 1.18], gap="small")
    with inventory_left:
        with st.container(border=True):
            st.subheader("▦  Recorded stock distribution")
            st.caption("SKU count by recorded stock band; 51+ retains all higher source values")
            if not stock_band_counts.empty:
                stock_plot = stock_band_counts[stock_band_counts["Stock band"] != "Missing"].copy()
                stock_colors = {"0": RED, "1-10": AMBER, "11-25": BLUE, "26-50": BLUE, "51+": GREEN}
                stock_chart = px.bar(
                    stock_plot,
                    x="Stock band",
                    y="SKUs",
                    text="SKUs",
                    title=None,
                    color="Stock band",
                    color_discrete_map=stock_colors,
                    hover_data={"SKUs": True},
                    category_orders={"Stock band": ["0", "1-10", "11-25", "26-50", "51+"]},
                )
                stock_chart.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
                stock_chart.update_layout(height=255, margin=dict(l=8, r=12, t=14, b=8), bargap=.3)
                stock_chart.update_xaxes(title="Stock level", type="category", showgrid=False)
                stock_chart.update_yaxes(title="SKUs", showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
                _chart(stock_chart, height=255)
                st.caption(f"Missing stock value: {missing_stock}. Values above 50 are grouped for readability, not recoded.")
            else:
                st.info("No inventory stock values available.")
    with inventory_right:
        with st.container(border=True):
            st.subheader("☷  Low/zero stock products (sample)")
            st.caption("Priority list: critical zero stock, positive low stock, and missing stock")
            attention = inventory[inventory["Stock"].isna() | inventory["Stock"].le(10)].copy()
            attention["Alert status"] = np.select(
                [attention["Stock"].isna(), attention["Stock"].eq(0)],
                ["DATA QUALITY: Missing stock", "CRITICAL: Zero stock"],
                default="WARNING: 1-10 units",
            )
            attention = attention.sort_values("Stock", na_position="first")
            if not attention.empty:
                display_attention = attention[["SKU", "Product_Name", "Stock", "Alert status", "Unit_Price_Currency", "Unit_Price_Amount"]].rename(
                    columns={"Product_Name": "Product name", "Unit_Price_Currency": "Currency", "Unit_Price_Amount": "Unit price"}
                ).head(12)
                st.dataframe(display_attention, width="stretch", hide_index=True, height=270)
            else:
                st.info("No products meet the zero/low-stock review thresholds.")

    st.info("Inventory thresholds are operational indicators. Transactions have no SKU key, so this page cannot estimate lost sales or demand-based reorder quantities. Marketing has no reliable order link, campaign cost, or conversion fields, so CTR is not ROI.")


def _decision_tables(orders: pd.DataFrame, customers: pd.DataFrame, marketing: pd.DataFrame, inventory: pd.DataFrame, thresholds: dict[str, Any], benchmark_marketing: pd.DataFrame | None = None, model_results: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = inventory["Stock"].dropna()
    zero_count = int(stock.eq(0).sum())
    low_count = int((stock.gt(0) & stock.le(10)).sum())
    missing_stock_count = int(inventory["Stock"].isna().sum())
    high_value_count = int((customers["Customer_Total_Revenue"] > thresholds["revenue_p75"]).sum()) if not customers.empty else 0
    low_frequency_count = int((customers["Customer_Purchase_Count"].gt(0) & customers["Customer_Frequency"].le(thresholds["frequency_p25"])).sum()) if not customers.empty else 0
    repeat_count = int(customers["Customer_Type"].eq("Repeat").sum()) if not customers.empty else 0
    risk_count = int(customers["Customer_Segment"].eq("Recency Review (Limited Data)").sum()) if not customers.empty else 0
    states = _state_summary(orders)
    named_states = states[states["State"] != "Unassigned"]
    state_median = float(named_states["Revenue"].median()) if not named_states.empty else 0.0
    priority_states = named_states.loc[named_states["Revenue"] > state_median, "State"].tolist()
    channel_source = benchmark_marketing if benchmark_marketing is not None else marketing
    channel_ctr = channel_source.dropna(subset=["CTR"]).groupby("Channel")["CTR"].mean()
    channel_median = float(channel_ctr.median()) if not channel_ctr.empty else 0.0
    low_ctr_channels = channel_ctr[channel_ctr < channel_median].index.tolist()
    linear_r2 = None
    if model_results is not None and not model_results.empty:
        linear_rows = model_results.loc[model_results["Model"] == "Linear Regression", "Test R²"]
        if not linear_rows.empty:
            linear_r2 = float(linear_rows.iloc[0])

    rules = pd.DataFrame(
        [
            {"Priority": "HIGH", "Condition": f"Customer revenue > P75 ({_currency(thresholds['revenue_p75'])})", "Status": f"{high_value_count} high-value customers", "Recommended action": "Prioritise retention and personalised-offer tests; measure incremental margin."},
            {"Priority": "MEDIUM", "Condition": "Purchase count > 1", "Status": f"{repeat_count} repeat customers", "Recommended action": "Use a loyalty/retention test and compare against a holdout."},
            {"Priority": "MEDIUM", "Condition": f"Purchase frequency <= P25 ({thresholds['frequency_p25']:.0f} purchase)", "Status": f"{low_frequency_count} low-frequency customers", "Recommended action": "Test an activation or replenishment reminder; evaluate incremental repeat orders."},
            {"Priority": "INFORMATION", "Condition": "At-risk classification limited by missing dates and short observation window", "Status": f"{risk_count} customers in recency review segment; not an at-risk label", "Recommended action": "Improve date completeness and history before using churn or re-engagement rules."},
            {"Priority": "CRITICAL", "Condition": "Stock = 0", "Status": f"{zero_count} zero-stock SKUs", "Recommended action": "Verify availability and trigger replenishment review; pause promotion until supply is confirmed."},
            {"Priority": "HIGH", "Condition": "0 < Stock <= 10 units", "Status": f"{low_count} low-stock SKUs", "Recommended action": "Review replenishment against demand velocity and supplier lead time."},
            {"Priority": "DATA QUALITY", "Condition": "Stock is missing", "Status": f"{missing_stock_count} inventory record(s)", "Recommended action": "Validate source count and complete the inventory record before availability decisions."},
            {"Priority": "HIGH", "Condition": f"Named-state revenue > median ({_currency(state_median)})", "Status": ", ".join(priority_states) if priority_states else "No state above benchmark", "Recommended action": "Review market capacity and fulfilment before increasing regional spend."},
            {"Priority": "MEDIUM", "Condition": f"Mean channel CTR < median ({channel_median:.2%})", "Status": ", ".join(low_ctr_channels) if low_ctr_channels else "No channel below benchmark", "Recommended action": "Review creative and audience; evaluate conversion and cost before reallocating spend."},
            {"Priority": "MODEL GOVERNANCE", "Condition": f"Test R² <= 0 (current Linear Regression R² {linear_r2:.3f})" if linear_r2 is not None else "Test R² <= 0", "Status": "BASELINE / NOT PRODUCTION READY" if linear_r2 is None or linear_r2 <= 0 else "Review model status", "Recommended action": "Do not automate forecasts; improve data and validate alternatives before reconsidering deployment."},
        ]
    )

    summary_rows = []
    if not named_states.empty:
        lead = named_states.iloc[0]
        summary_rows.append({"Pattern": "High-revenue state", "Evidence": f"{lead['State']}: {_currency(float(lead['Revenue']))} ({lead['Contribution']:.1%} of observed revenue)", "Business meaning": "Largest named-state revenue base in this selection.", "Recommended action": "Prioritise capacity and customer-experience review; monitor unresolved state attribution."})
    if not customers.empty:
        repeat_sales = customers.loc[customers["Customer_Type"].eq("Repeat"), "Customer_Total_Revenue"].sum(min_count=1)
        all_customer_sales = customers["Customer_Total_Revenue"].sum(min_count=1)
        repeat_share = float(repeat_sales / all_customer_sales) if pd.notna(all_customer_sales) and all_customer_sales else 0.0
        summary_rows.append({"Pattern": "Repeat customers", "Evidence": f"{repeat_count} customers; {repeat_share:.1%} of identified customer revenue", "Business meaning": "Repeat buyers are a material observed cohort, not proof of loyalty-program impact.", "Recommended action": "Test retention offers with a control group and track incremental margin."})
        summary_rows.append({"Pattern": "High-value customers", "Evidence": f"{high_value_count} customers above revenue P75 ({_currency(thresholds['revenue_p75'])})", "Business meaning": "Revenue is concentrated among customers above the observed cohort threshold.", "Recommended action": "Prioritise retention and service recovery for this segment."})
        high_frequency = int(customers["Customer_Frequency"].gt(thresholds["frequency_p75"]).sum())
        summary_rows.append({"Pattern": "High-frequency customers", "Evidence": f"{high_frequency} customers above frequency P75 ({thresholds['frequency_p75']:.0f} purchases)", "Business meaning": "These customers purchase more often than the cohort's upper-quartile boundary.", "Recommended action": "Review repeat-order needs and test a loyalty treatment against a holdout."})
        low_frequency = int((customers["Customer_Purchase_Count"].gt(0) & customers["Customer_Frequency"].le(thresholds["frequency_p25"])).sum())
        summary_rows.append({"Pattern": "Low-frequency customers", "Evidence": f"{low_frequency} customers at or below frequency P25 ({thresholds['frequency_p25']:.0f} purchase)", "Business meaning": "This group has the lowest observed purchase counts in the current cohort.", "Recommended action": "Test a relevant activation or reminder and measure incremental repeat purchases."})
    summary_rows.append({"Pattern": "Inventory risk", "Evidence": f"{zero_count} zero-stock; {low_count} positive stock <=10; {missing_stock_count} missing", "Business meaning": "Current stock snapshots identify availability and data-quality review candidates.", "Recommended action": "Verify zero/missing records, then review replenishment against lead time and demand."})
    if len(channel_ctr := channel_source.dropna(subset=["CTR"]).groupby("Channel")["CTR"].mean()):
        best_channel = channel_ctr.idxmax()
        worst_channel = channel_ctr.idxmin()
        summary_rows.append({"Pattern": "High-CTR channel", "Evidence": f"{best_channel}: {channel_ctr.max():.2%} mean CTR", "Business meaning": "Highest click engagement among observed channels; ROI is unavailable.", "Recommended action": "Run a controlled budget/creative test and validate conversions and cost."})
        summary_rows.append({"Pattern": "Low-CTR channel", "Evidence": f"{worst_channel}: {channel_ctr.min():.2%} mean CTR", "Business meaning": "Lowest click engagement among observed channels.", "Recommended action": "Review message, audience, and placement; do not infer sales impact."})
    observed_sales = orders["Total_Sales"].dropna()
    if not observed_sales.empty:
        q1 = float(observed_sales.quantile(0.25))
        q3 = float(observed_sales.quantile(0.75))
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        low_outliers = int(observed_sales.lt(low_fence).sum())
        high_outliers = int(observed_sales.gt(high_fence).sum())
        summary_rows.append({"Pattern": "Transaction-value outliers", "Evidence": f"IQR fences ${low_fence:,.2f} to ${high_fence:,.2f}; {low_outliers} low / {high_outliers} high flags", "Business meaning": "No flagged values should be treated as errors without source review.", "Recommended action": "Investigate future flags against source records, returns, and promotions before excluding them."})
    monthly = _monthly_summary(orders)
    observed_months = monthly.dropna(subset=["Revenue"])
    if not observed_months.empty:
        best = observed_months.loc[observed_months["Revenue"].idxmax()]
        summary_rows.append({"Pattern": "Highest dated month", "Evidence": f"{best['Month']}: {_currency(float(best['Revenue']))} across {int(best['Observed_Sales_Transactions'])} dated orders with observed sales", "Business meaning": "Largest observed month bucket; date completeness is limited.", "Recommended action": "Validate date coverage and repeat the comparison on complete periods."})
    return rules, pd.DataFrame(summary_rows)


def _show_ethics_governance(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    marketing: pd.DataFrame,
    context: dict[str, Any],
) -> None:
    _header("Ethics, Governance & Responsible Analytics")
    st.caption("Ensure data is used ethically, fairly, and responsibly to build trustworthy insights and support sound business decisions.")
    quality = context["quality"]
    inventory = context["inventory"]
    missing_ids = quality["transaction_customer_ids_missing"]
    missing_dates = quality["order_dates_missing"]
    missing_sales = quality["sales_missing"]
    unassigned_orders = int(orders["State"].eq("Unassigned").sum())
    source_audit = quality["source_reconciliation"]

    with st.container(border=True):
        st.subheader("⚖  Ethical principles and project application")
        deontology, utility, virtue = st.columns(3, gap="small")
        principle_cards = [
            (
                deontology,
                "📄  Deontology",
                "Do what is right",
                [
                    "Protect customer privacy and sensitive information.",
                    "Report results honestly, including uncertainty and missing data.",
                    "Avoid unsupported claims or misleading insights.",
                ],
                "#fff0f2",
                "#f3d8df",
            ),
            (
                utility,
                "👥  Utilitarianism",
                "Maximise overall good",
                [
                    "Use data to create sustainable value for BlueCart and its customers.",
                    "Minimise harm from inaccurate forecasts or poor stock decisions.",
                    "Balance commercial benefit with privacy, fairness, and oversight.",
                ],
                "#edf9f2",
                "#d0eedb",
            ),
            (
                virtue,
                "🛡  Virtue Ethics",
                "Act with integrity",
                [
                    "Demonstrate honesty, fairness, and transparency.",
                    "Apply professional judgement and analytical competence.",
                    "Take responsibility for the impact of data-driven decisions.",
                ],
                "#f4f1ff",
                "#e3dcfb",
            ),
        ]
        for column, title, subtitle, bullets, fill, border in principle_cards:
            with column:
                st.markdown(
                    f"<div style='background:{fill};border:1px solid {border};border-radius:9px;padding:.8rem .9rem;min-height:178px'>"
                    f"<div style='font-weight:800;color:#0b1645'>{title}</div>"
                    f"<div style='font-size:.76rem;color:#68758a;margin:.15rem 0 .45rem'>{subtitle}</div>"
                    + "".join(f"<div style='font-size:.75rem;line-height:1.35;margin:.3rem 0'>• {bullet}</div>" for bullet in bullets)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    privacy, bias = st.columns([1.04, 1.3], gap="small")
    with privacy:
        with st.container(border=True):
            st.subheader("🔒  Privacy, fairness & responsible use")
            topics = st.columns(2)
            topic_items = [
                ("🔐 Privacy", "Names, emails, and direct identifiers are excluded from dashboard views and model features. Customer-level analysis uses masked IDs."),
                ("▤ Data minimisation", "Use only fields needed for the stated analysis; restrict access to identifiable source workbooks."),
                ("⚖ Fairness", "Do not infer sensitive traits or automatically include/exclude customers based on incomplete history."),
                ("👤 Human review", "Rules flag records for review; authorised people remain responsible for high-impact actions."),
            ]
            for index, (label, body) in enumerate(topic_items):
                with topics[index % 2]:
                    st.markdown(
                        f"<div style='background:#f8faff;border:1px solid #e0e8f4;border-radius:8px;padding:.55rem .6rem;min-height:102px;margin-bottom:.45rem'>"
                        f"<strong style='color:#164477;font-size:.78rem'>{label}</strong><div style='font-size:.72rem;line-height:1.3;margin-top:.25rem'>{body}</div></div>",
                        unsafe_allow_html=True,
                    )
    with bias:
        with st.container(border=True):
            st.subheader("⚠  Bias, data limitations & risks")
            risk_cols = st.columns(5)
            risk_cols[0].metric("Missing order dates", f"{missing_dates:,}")
            risk_cols[1].metric("Missing customer IDs", f"{missing_ids:,}")
            risk_cols[2].metric("Unassigned state orders", f"{unassigned_orders:,}")
            risk_cols[3].metric("No SKU link", "Yes")
            risk_cols[4].metric("No campaign link", "Yes")
            st.markdown(
                "<div style='background:#fff0f0;border:1px solid #f2cccc;border-radius:8px;padding:.55rem .7rem;font-size:.75rem;color:#71302a;margin:.35rem 0'>"
                "<strong>Risk:</strong> Missing or disconnected data can create misleading patterns, inaccurate segments, and unfair prioritisation. The alternate raw CSV also conflicts with the validated DataVault on 75 sales rows; use the cleaned workbook and investigate discrepancies before operational use."
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='background:#eaf8ef;border:1px solid #ccebd5;border-radius:8px;padding:.55rem .7rem;font-size:.75rem;color:#24533a'>"
                "<strong>Mitigation:</strong> report completeness, validate joins and source versions, use transparent segment definitions, monitor subgroup outcomes, and require human review for consequential actions."
                "</div>",
                unsafe_allow_html=True,
            )

    acs_col, governance_col = st.columns([1, 1.35], gap="small")
    with acs_col:
        with st.container(border=True):
            st.subheader("▣  ACS professional responsibilities")
            acs_items = [
                ("Public Interest", "Protect privacy and avoid harmful or unfair automated decisions."),
                ("Honesty", "Report nonsignificant tests, missing data, source conflicts, and weak models without exaggeration."),
                ("Competence", "Validate data preparation, joins, calculations, and models before advising action."),
                ("Professionalism", "Document assumptions and changes; maintain accountability and human oversight."),
            ]
            acs_cols = st.columns(2)
            acs_colors = ["#edf6ff", "#fff8e8", "#edf9f2", "#f4f1ff"]
            for index, (principle, statement) in enumerate(acs_items):
                with acs_cols[index % 2]:
                    st.markdown(
                        f"<div style='background:{acs_colors[index]};border:1px solid #dfe5ee;border-radius:8px;padding:.55rem .62rem;min-height:87px;margin-bottom:.45rem'>"
                        f"<strong style='font-size:.77rem;color:#0b1645'>{principle}</strong><div style='font-size:.71rem;line-height:1.3;margin-top:.2rem'>{statement}</div></div>",
                        unsafe_allow_html=True,
                    )
            st.caption("ACS Code application: keep public/customer interests ahead of convenience; communicate evidence faithfully; use methods within competence; own and document professional decisions.")
    with governance_col:
        with st.container(border=True):
            st.subheader("🏛  Governance and accountability framework")
            owners = [
                ("Data owner", "Operations\nSales definitions"),
                ("Data steward", "CRM / inventory\nQuality & access"),
                ("Analyst", "Prepare, analyse\nDocument models"),
                ("Manager", "Review insights\nApprove actions"),
                ("Assurance", "Audit changes\nReview compliance"),
            ]
            owner_cols = st.columns(5)
            for index, (role, responsibility) in enumerate(owners):
                with owner_cols[index]:
                    st.markdown(
                        f"<div style='background:#f5f8fd;border:1px solid #dce5f1;border-radius:7px;padding:.48rem .4rem;min-height:84px;text-align:center'>"
                        f"<strong style='font-size:.72rem;color:#0b1645'>{role}</strong><div style='font-size:.66rem;line-height:1.25;margin-top:.24rem;white-space:pre-line'>{responsibility}</div></div>",
                        unsafe_allow_html=True,
                    )
                if index < len(owners) - 1:
                    st.markdown("<div style='display:none'>→</div>", unsafe_allow_html=True)
            control_cols = st.columns(5)
            controls = [
                ("🔒", "Role-based\naccess"),
                ("▤", "Data quality\nchecks"),
                ("⌘", "Versioned\nmodels"),
                ("◷", "Change\nlog"),
                ("👤", "Human approval\nfor high impact"),
            ]
            for column, (icon, label) in zip(control_cols, controls):
                with column:
                    st.markdown(f"<div style='text-align:center;font-size:.67rem;color:#425574;padding:.35rem 0'>{icon}<br>{label}</div>", unsafe_allow_html=True)
            st.caption("Management remains accountable for decisions. Before implementation, assign named owners and have the privacy/legal owner review applicable Privacy Act and Australian Privacy Principles obligations, consent, retention, and access policy.")

    data_col, action_col = st.columns([1, 1.2], gap="small")
    with data_col:
        with st.container(border=True):
            st.subheader("▥  Data quality overview")
            st.caption("Missing values in the cleaned analytical tables")
            field_missing = quality["field_missing"]
            chosen_fields = [
                ("Order date", "Transactions", "Order_Date"),
                ("Customer ID", "Transactions", "Customer_ID_Masked"),
                ("Order value", "Transactions", "Total_Sales"),
                ("Campaign date", "Marketing", "Campaign_Date"),
                ("Campaign CTR", "Marketing", "CTR"),
                ("Inventory stock", "Inventory", "Stock"),
            ]
            missing_rows = []
            for display, table_name, field_name in chosen_fields:
                match = field_missing[(field_missing["Table"] == table_name) & (field_missing["Field"] == field_name)]
                count = int(match["Missing"].iloc[0]) if not match.empty else 0
                missing_rows.append({"Field": display, "Missing": count})
            missing_plot = pd.DataFrame(missing_rows).sort_values("Missing")
            missing_chart = px.bar(missing_plot, x="Missing", y="Field", orientation="h", text="Missing", title=None, color="Missing", color_continuous_scale=["#66b5ee", ORANGE])
            missing_chart.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
            missing_chart.update_layout(height=265, margin=dict(l=8, r=20, t=10, b=8), coloraxis_showscale=False)
            missing_chart.update_xaxes(title="Missing values", showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
            missing_chart.update_yaxes(title=None, tickfont=dict(size=10))
            _chart(missing_chart, height=265)
    with action_col:
        with st.container(border=True):
            st.subheader("💡  Key takeaways and recommendations")
            recommendations = [
                "Protect customer privacy; minimise identifiers and restrict access to authorised stewards.",
                f"Address {missing_dates} missing order dates and {missing_ids} missing customer IDs to reduce bias and improve analysis coverage.",
                "Use transparent, validated segments and explainable rules; do not assign churn status from this short incomplete history.",
                "Require human review for customer targeting, inventory replenishment, and all high-impact actions.",
                "Track source, dashboard, rule, and model changes; repeat completeness and model-performance checks before release.",
            ]
            takeaway_html = ""
            badge_colors = [BLUE, ORANGE, AMBER, "#7856c5", GREEN]
            for index, item in enumerate(recommendations, start=1):
                takeaway_html += (
                    f"<div class='bc-finding'><span class='bc-finding-num' style='background:{badge_colors[index-1]};color:#fff'>{index}</span>"
                    f"<span>{item}</span></div>"
                )
            st.markdown(takeaway_html, unsafe_allow_html=True)
            st.markdown(
                "<div style='background:#e9f8ef;border:1px solid #c6ead3;border-radius:8px;padding:.52rem .7rem;font-size:.74rem;color:#1c5234;margin-top:.45rem'><strong>Responsible decision principle:</strong> Use transparent, evidence-based rules with human review. Do not automate forecasting, customer decisions, or budget attribution until data quality and cross-table linkage are validated.</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Detailed evidence, source reconciliation, and Sprint 1 references"):
        st.subheader("Field-level missingness")
        st.dataframe(field_missing.assign(Percent=lambda frame: (frame["Missing"] / frame["Rows"] * 100).map(lambda value: f"{value:.1f}%")), width="stretch", hide_index=True)
        st.subheader("Validated table relationships")
        st.dataframe(quality["relationships"], width="stretch", hide_index=True)
        if source_audit["raw_workbook_available"] and source_audit["alternate_csv_sales_disagreements"]:
            st.warning(
                f"The cleaned workbook matches {source_audit['matched_order_ids']} DataVault order IDs and removes {source_audit['nonpositive_sales_removed']} nonpositive sales. "
                f"The alternate raw CSV disagrees on {source_audit['alternate_csv_sales_disagreements']} sales values, including {source_audit['alternate_csv_positive_for_raw_nonpositive']} rows positive in that CSV but nonpositive in the DataVault. Use the cleaned workbook; do not combine the alternate CSV."
            )
        st.subheader("Sprint 1 cleaning summary")
        st.dataframe(context["summary"].fillna("").astype(str), width="stretch", hide_index=True)
        st.subheader("Data dictionary")
        st.dataframe(context["dictionary"], width="stretch", hide_index=True)

    with st.expander("Professional-practice detail: frameworks, alternatives, recommendations, and rubric"):
        st.markdown(
            "**Ethical frameworks:** Deontology requires protecting privacy and reporting findings honestly; Utilitarianism balances BlueCart benefit against potential customer and operational harm; Virtue Ethics requires honesty, fairness, care, competence, and transparency.\n\n"
            "**ACS Code application:** Primacy of the Public Interest constrains targeting where privacy/fairness may be harmed. Honesty requires clear reporting of nonsignificant tests, missingness, source conflict, and weak model performance. Competence requires validating data, joins, and models. Professionalism requires documenting assumptions, access, changes, and accountability.\n\n"
            "**Governance controls:** assign data owner/steward/analyst/manager roles; use role-based access; schedule data-quality checks; version datasets, models, and thresholds; monitor performance and drift; keep alert reason codes; require human approval; and retain an audit/change log. Management remains accountable. Have the responsible privacy/legal owner review applicable Privacy Act 1988 and Australian Privacy Principles requirements, consent, and retention before implementation.\n\n"
            "**Alternatives:** tested Linear Regression, Decision Tree, and Random Forest on the same 80/20 split; current test R² values are negative. Gradient Boosting and K-Means remain future candidates only after more complete data and validation. Bar/line/scatter charts are used for comparisons, time, and relationships; a heatmap is deferred because the period coverage is sparse."
        )
        st.subheader("Recommended actions with owners and risks")
        recommendation_rows = [
            ["Retention pilot", "CRM lead", "Test high-value customer offers against a holdout.", "Over-targeting / incomplete identity; measure incremental margin."],
            ["Complete source linkage", "Data stewards", "Add complete IDs/dates, SKU/order, and campaign/order keys.", "Inconsistent definitions; require schema ownership and validation."],
            ["Inventory review", "Inventory control", "Verify zero and missing stock, then review low-stock reorder needs.", "No SKU sales linkage; do not infer lost sales."],
            ["Model readiness", "Analytics lead + manager", "Keep forecasts advisory; improve data and time-aware validation.", "Complex models may overfit; management approves all use."],
        ]
        recommendation_table = pd.DataFrame(recommendation_rows, columns=["Action", "Proposed owner", "Next step", "Risk / control"])
        st.dataframe(recommendation_table, width="stretch", hide_index=True)
        rubric_coverage = pd.DataFrame(
            [
                ["Executive summary", "Decision page takeaways and recommended actions"],
                ["Visual/design quality", "KPI cards, chart/table hierarchy, text labels, colour-coded status plus text"],
                ["Storytelling", "Evidence → limitation → policy/control → action"],
                ["Decision support", "Visible rules, thresholds, alert counts, human review"],
                ["Alternatives/evaluation", "Tested regressors, future methods, chart rationale"],
                ["Ethics/professionalism", "Privacy, fairness, bias, ethical frameworks, ACS principles"],
                ["Actions/risks", "Named proposed owners, risks, controls, next data needs"],
            ],
            columns=["Rubric area", "Visible evidence"],
        )
        st.subheader("Sprint 3 rubric coverage")
        st.dataframe(rubric_coverage, width="stretch", hide_index=True)
def _show_decision_support(orders: pd.DataFrame, customers: pd.DataFrame, marketing: pd.DataFrame, context: dict[str, Any]) -> None:
    _header("BlueCart Decision Support & Alerts")
    rules, patterns = _decision_tables(orders, customers, marketing, context["inventory"], context["thresholds"], context["all_channels_marketing"], context["model_results"])
    quality = context["quality"]
    inventory = context["inventory"].copy()
    stock = inventory["Stock"]
    zero_stock = int(stock.eq(0).sum())
    low_stock = int((stock.gt(0) & stock.le(10)).sum())
    missing_stock = int(stock.isna().sum())
    total_transactions = int(orders["Order_ID"].nunique(dropna=True))
    observed_customer_count = int((customers["Customer_Purchase_Count"] > 0).sum()) if not customers.empty else 0
    named_state = _state_summary(orders)
    named_state = named_state[named_state["State"] != "Unassigned"]
    state_lead = named_state.iloc[0] if not named_state.empty else None
    top_state = state_lead["State"] if state_lead is not None else "no named state"
    repeat_cohort = context["statistical_comparisons"]["customer_welch"]
    channel_ctrs = context["all_channels_marketing"].dropna(subset=["CTR"]).groupby("Channel")["CTR"].mean().sort_values(ascending=False)
    best_channel = channel_ctrs.index[0] if not channel_ctrs.empty else "No channel"
    best_channel_ctr = float(channel_ctrs.iloc[0]) if not channel_ctrs.empty else 0.0
    model_results = context["model_results"]
    linear_result = model_results.loc[model_results["Model"] == "Linear Regression"].iloc[0]
    tree_result = model_results.loc[model_results["Model"] == "Decision Tree"].iloc[0]

    st.markdown("**Decision logic:** observed quartiles define customer and market review groups; stock uses explicit 0 and 1-10 unit alerts; channel review uses the observed channel-median CTR. These are transparent review triggers, not automated business decisions.")
    with st.container(border=True):
        rule_title, rule_status = st.columns([.8, .2], vertical_alignment="center")
        with rule_title:
            st.subheader("🔔  Decision rules and alerts")
            st.caption("Conditions, current status, and recommended human action")
        with rule_status:
            st.markdown("<div style='text-align:right;background:#e8f7ee;border:1px solid #c8ebd5;border-radius:8px;padding:.45rem .6rem;color:#17643d;font-weight:700'>✓ 10 rules configured</div>", unsafe_allow_html=True)
        st.dataframe(rules.rename(columns={"Priority": "Priority", "Condition": "Condition", "Status": "Current status", "Recommended action": "Recommended action"}), width="stretch", hide_index=True, height=330)

    mined, takeaways = st.columns([1.45, 1], gap="small")
    with mined:
        with st.container(border=True):
            st.subheader("▱  Data-mining summary")
            st.caption("Validated data scope underlying the current rule set")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Transaction rows", f"{quality['transaction_rows']:,}", help=f"{quality['duplicate_order_ids']} duplicate order IDs.")
            metric_cols[1].metric("Customers with sales", f"{observed_customer_count:,}", help=f"{quality['transaction_customer_ids_missing']} transaction customer IDs are missing; additional IDs have no valid sales.")
            metric_cols[2].metric("Campaigns", f"{int(context['all_channels_marketing']['Campaign_ID'].nunique()):,}")
            metric_cols[3].metric("SKUs", f"{int(inventory['SKU'].nunique()):,}")
            st.caption(f"Observed positive sales: {_currency(float(orders['Total_Sales'].sum(skipna=True)))} across {int(orders['Total_Sales'].notna().sum())} transactions; the many-to-one customer join preserves unique order cardinality.")
    with takeaways:
        state_takeaway = (
            f"{state_lead['State']} leads named-state revenue at USD {float(state_lead['Revenue']):,.2f}."
            if state_lead is not None
            else "No named-state leader is available in this selection."
        )
        repeat_takeaway = (
            f"Repeat buyers account for {repeat_cohort['repeat_n']} observed transactions vs {repeat_cohort['one_time_n']} one-time; mean-value difference is not significant (p={repeat_cohort['p_value']:.3f})."
            if repeat_cohort["p_value"] is not None
            else "Customer-type comparison unavailable in this selection."
        )
        takeaway_items = [
            state_takeaway,
            repeat_takeaway,
            f"Inventory snapshot: {zero_stock} zero-stock and {low_stock} positive low-stock items; not linked to sales by SKU.",
            f"{best_channel} has the highest mean CTR ({best_channel_ctr:.2%}); CTR is engagement, not ROI.",
        ]
        takeaway_markup = "<div class='bc-key-findings' style='background:#edf5ff;border-color:#d5e5fa;color:#1b3556'><h3 style='color:#143e6e'>💡 Key takeaways</h3>"
        takeaway_markup += "".join(
            f"<div class='bc-finding'><span class='bc-finding-num' style='background:#deebfc;color:#1d5da2'>{index}</span><span>{item}</span></div>"
            for index, item in enumerate(takeaway_items, start=1)
        )
        takeaway_markup += "</div>"
        st.markdown(takeaway_markup, unsafe_allow_html=True)

    quality_col, limits_col = st.columns([1.45, 1], gap="small")
    with quality_col:
        with st.container(border=True):
            st.subheader("🛡  Data quality & limitations")
            qcols = st.columns(4)
            qcols[0].metric("Missing order dates", f"{quality['order_dates_missing']:,}")
            qcols[1].metric("Missing customer IDs", f"{quality['transaction_customer_ids_missing']:,}")
            qcols[2].metric("Missing sales", f"{quality['sales_missing']:,}")
            qcols[3].metric("Duplicate Order IDs", f"{quality['duplicate_order_ids']:,}")
            missing = quality["field_missing"].copy()
            if not missing.empty:
                missing["% missing"] = (missing["Missing"] / missing["Rows"] * 100).map(lambda value: f"{value:.1f}%")
                st.dataframe(missing[["Table", "Field", "Missing", "% missing"]], width="stretch", hide_index=True, height=210)
            st.caption("Customer join is validated many-to-one. Marketing has no transaction Campaign_ID, and transactions have no SKU/Product ID.")
    with limits_col:
        with st.container(border=True):
            st.subheader("⚠  Key limitations")
            limitations = [
                f"{quality['order_dates_missing']} missing dates limit temporal analysis; March has only 3 dated observed sales.",
                "Missing customer IDs limit customer attribution and segment coverage.",
                f"Marketing has {quality['marketing_dates_missing']} missing dates, {quality['marketing_reach_missing']} missing reach values, and {quality['marketing_ctr_missing']} missing CTR values.",
                "Campaign cost/conversion and order campaign keys are unavailable; do not infer ROI or causality.",
                "No SKU key on transactions; stock cannot be linked to sales or lost revenue.",
                "Customer recency is a limited review signal, not a churn/at-risk prediction.",
            ]
            limitation_html = "<div style='padding:.15rem .2rem;color:#5a4611'>"
            limitation_html += "".join(f"<div style='display:flex;gap:.5rem;margin:.42rem 0;font-size:.79rem;line-height:1.3'><b style='color:#d58a00'>•</b><span>{item}</span></div>" for item in limitations)
            limitation_html += "</div>"
            st.markdown(limitation_html, unsafe_allow_html=True)

    source_audit = quality["source_reconciliation"]
    if source_audit["raw_workbook_available"] and source_audit["alternate_csv_sales_disagreements"]:
        st.warning(
            f"Source reconciliation: the cleaned workbook matches all {source_audit['matched_order_ids']} DataVault order IDs and removes "
            f"{source_audit['nonpositive_sales_removed']} nonpositive source sales. The alternate raw CSV disagrees on "
            f"{source_audit['alternate_csv_sales_disagreements']} order values, including "
            f"{source_audit['alternate_csv_positive_for_raw_nonpositive']} rows that are nonpositive in the DataVault but positive in that CSV. "
            "This dashboard uses the validated cleaned workbook; do not combine the conflicting alternate CSV."
        )

    stock_bands = inventory.copy()
    stock_bands["Stock band"] = np.select(
        [stock_bands["Stock"].isna(), stock_bands["Stock"].eq(0), stock_bands["Stock"].between(1, 10), stock_bands["Stock"].between(11, 25), stock_bands["Stock"].between(26, 50)],
        ["Missing", "0", "1-10", "11-25", "26-50"],
        default="51+",
    )
    stock_order = ["Missing", "0", "1-10", "11-25", "26-50", "51+"]
    stock_counts = stock_bands["Stock band"].value_counts().reindex(stock_order, fill_value=0).rename_axis("Stock band").reset_index(name="SKUs")
    stock_counts = stock_counts[stock_counts["SKUs"].gt(0) & stock_counts["Stock band"].ne("Missing")]
    stock_list = inventory[inventory["Stock"].isna() | inventory["Stock"].le(10)].copy()
    stock_list["Alert status"] = np.select([stock_list["Stock"].isna(), stock_list["Stock"].eq(0)], ["DATA QUALITY: Missing stock", "CRITICAL: Zero stock"], default="WARNING: 1-10 units")
    stock_list = stock_list.sort_values("Stock", na_position="first").head(10)
    stock_left, stock_right = st.columns([1, 1.38], gap="small")
    with stock_left:
        with st.container(border=True):
            st.subheader("▦  Recorded stock distribution")
            st.caption("SKU count by recorded stock level; 51+ retains higher source values")
            if not stock_counts.empty:
                stock_chart = px.bar(stock_counts, x="Stock band", y="SKUs", text="SKUs", color="Stock band", title=None, color_discrete_map={"0": RED, "1-10": AMBER, "11-25": BLUE, "26-50": BLUE, "51+": GREEN}, category_orders={"Stock band": ["0", "1-10", "11-25", "26-50", "51+"]})
                stock_chart.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
                stock_chart.update_layout(height=250, margin=dict(l=8, r=12, t=14, b=8), bargap=.3)
                stock_chart.update_xaxes(title="Stock level", type="category", showgrid=False)
                stock_chart.update_yaxes(title="SKUs", showgrid=True, gridcolor="#edf1f6", rangemode="tozero")
                _chart(stock_chart, height=250)
            st.caption(f"CRITICAL {zero_stock} | WARNING {low_stock} | MISSING {missing_stock}. Missing values are reported separately.")
    with stock_right:
        with st.container(border=True):
            st.subheader("☷  Low/zero stock product list (sample)")
            st.caption("CRITICAL = 0 units · WARNING = 1-10 units · DATA QUALITY = missing")
            st.dataframe(stock_list[["SKU", "Product_Name", "Stock", "Alert status", "Unit_Price_Currency", "Unit_Price_Amount"]].rename(columns={"Product_Name": "Product name", "Unit_Price_Currency": "Currency", "Unit_Price_Amount": "Unit price"}), width="stretch", hide_index=True, height=250)

    model_results = context["model_results"]
    linear_result = model_results.loc[model_results["Model"] == "Linear Regression"].iloc[0]
    tree_result = model_results.loc[model_results["Model"] == "Decision Tree"].iloc[0]
    with st.container(border=True):
        st.subheader("📊  Predictive modelling context")
        st.caption("Sprint 2 regression is retained as supporting analysis; negative test R² means weak, advisory-only performance.")
        model_cols = st.columns(4)
        model_cols[0].metric("Linear Regression test R²", f"{linear_result['Test R²']:.3f}")
        model_cols[1].metric("Linear Regression RMSE", _currency(float(linear_result["Test RMSE"])))
        model_cols[2].metric("Decision Tree test R²", f"{tree_result['Test R²']:.3f}")
        model_cols[3].metric("Model status", "Weak baseline")
        st.caption("Sprint 2 reported Linear Regression test MAE $38.99 / RMSE $44.34 / R² -0.024 and Decision Tree test MAE $39.62 / RMSE $45.91 / R² -0.097. Current workbook replay follows below; both versions remain weak baselines.")
        st.dataframe(model_results.round(3), width="stretch", hide_index=True)
        actual_predictions = context["linear_predictions"]
        if not actual_predictions.empty:
            model_left, model_right = st.columns(2, gap="small")
            with model_left:
                actual_chart = px.scatter(actual_predictions, x="Actual", y="Predicted", title="Actual vs predicted sales (Linear Regression)", color_discrete_sequence=[BLUE])
                actual_chart.update_xaxes(title="Actual sales", tickprefix="$", showgrid=True, gridcolor="#edf1f6")
                actual_chart.update_yaxes(title="Predicted sales", tickprefix="$", showgrid=True, gridcolor="#edf1f6")
                _chart(actual_chart, height=245)
            with model_right:
                residuals = actual_predictions.assign(Residual=lambda frame: frame["Predicted"] - frame["Actual"])
                residual_chart = px.histogram(residuals, x="Residual", nbins=12, title="Residual distribution", color_discrete_sequence=["#8765d8"])
                residual_chart.update_xaxes(title="Prediction error", tickprefix="$", showgrid=True, gridcolor="#edf1f6")
                residual_chart.update_yaxes(title="Test transactions", showgrid=True, gridcolor="#edf1f6")
                _chart(residual_chart, height=245)

    welch_p = repeat_cohort["p_value"]
    welch_interpretation = (
        f"mean order value difference is not significant (p={welch_p:.3f})"
        if welch_p is not None
        else "customer-type comparison is unavailable in this small selection"
    )
    state_test = context["statistical_comparisons"]["state_anova"]
    state_interpretation = (
        f"ANOVA p={state_test['p_value']:.3f}"
        if state_test["p_value"] is not None
        else "state comparison unavailable in this small selection"
    )
    with st.expander("Recommended actions, expected value, and risks", expanded=False):
        recommendations = pd.DataFrame(
            [
                {"Priority": "1 | Retention", "Evidence / why": f"{int((customers['Customer_Total_Revenue'] > context['thresholds']['revenue_p75']).sum())} customers exceed revenue P75 ({_currency(context['thresholds']['revenue_p75'])}).", "Action": "Pilot personalised retention and loyalty treatments for high-value customers.", "Expected impact": "Potentially protect repeat revenue and customer lifetime value; measure with a holdout.", "Risk / owner": "Risk: over-targeting or missed customers from incomplete IDs. Owner: CRM / customer-data steward."},
                {"Priority": "2 | Repeat purchase", "Evidence / why": f"Repeat buyers contribute {repeat_cohort['repeat_n']} observed transactions vs {repeat_cohort['one_time_n']} one-time; {welch_interpretation}.", "Action": "Test a loyalty or replenishment reminder, evaluating repeat rate and margin rather than assuming higher spend.", "Expected impact": "Potentially increase repeat purchase incidence; no uplift assumed before experiment.", "Risk / owner": "Risk: treatment may not be incremental. Owner: CRM lead; require a control group."},
                {"Priority": "3 | Market review", "Evidence / why": f"{top_state} has the highest observed named-state revenue; {state_interpretation}.", "Action": "Review service capacity and customer experience before adjusting regional allocation.", "Expected impact": "Focus operational review on the largest observed revenue base.", "Risk / owner": "Risk: state mean differences are not significant and Unassigned revenue limits attribution. Owner: regional operations."},
                {"Priority": "4 | Inventory alerts", "Evidence / why": f"{zero_stock} zero stock, {low_stock} positive stock <=10, and {missing_stock} missing stock records.", "Action": "Investigate zero-stock SKUs, review low-stock reorder requirements, and complete missing records.", "Expected impact": "Improve availability oversight and inventory data completeness.", "Risk / owner": "Risk: sales cannot be linked to SKU; do not infer lost revenue. Owner: inventory control."},
                {"Priority": "5 | Marketing measurement", "Evidence / why": f"{quality['marketing_ctr_missing']} CTR and {quality['marketing_reach_missing']} reach values are missing; campaign attribution is unavailable.", "Action": "Capture Campaign_ID and acquisition source on orders, alongside spend and conversions.", "Expected impact": "Enable future conversion and incremental ROI evaluation.", "Risk / owner": "Risk: CTR-only allocation can waste budget. Owner: marketing analytics."},
                {"Priority": "6 | Model readiness", "Evidence / why": f"Linear Regression test R²={linear_result['Test R²']:.3f}; Decision Tree test R²={tree_result['Test R²']:.3f}.", "Action": "Keep forecasts advisory; improve feature coverage and use repeated/time-aware validation before any deployment.", "Expected impact": "Reduce forecast risk and improve evidence before operational use.", "Risk / owner": "Risk: added model complexity can overfit. Owner: analytics lead; management retains decision accountability."},
            ]
        )
        st.dataframe(recommendations, width="stretch", hide_index=True)

    with st.expander("Alternative methods and design rationale"):
        alternatives = pd.DataFrame(
            [
                {"Approach": "Linear Regression (current baseline)", "Strength / role": "Simple, interpretable transaction-level benchmark; current best of tested options by test RMSE.", "Limitation / decision": "Negative test R²; baseline only, not production-ready."},
                {"Approach": "Decision Tree (tested alternative)", "Strength / role": "Can represent nonlinear splits and interactions.", "Limitation / decision": f"Test RMSE {_currency(float(tree_result['Test RMSE']))}, worse than Linear Regression {_currency(float(linear_result['Test RMSE']))}; train/test gap suggests overfit risk."},
                {"Approach": "Random Forest (tested alternative)", "Strength / role": "Can capture nonlinear patterns with an ensemble.", "Limitation / decision": "Keep exploratory; limited sample and weak holdout performance require more data and cross-validation."},
                {"Approach": "Gradient Boosting (future candidate)", "Strength / role": "May improve nonlinear prediction on richer feature sets.", "Limitation / decision": "Not selected or tested here; validate only after improving data volume and leakage controls."},
                {"Approach": "RFM (implemented with guardrails)", "Strength / role": "Transparent segmentation using observed Recency, Frequency, and Monetary value.", "Limitation / decision": "Short, incomplete date window prevents a dependable churn/at-risk classification."},
                {"Approach": "K-Means (not applied)", "Strength / role": "Could explore unlabeled customer groups.", "Limitation / decision": "Needs stable numeric features, scaling, cluster validation, and enough observed customer history."},
                {"Approach": "Bar / line charts (selected)", "Strength / role": "Bars support exact category comparisons; lines show time order.", "Limitation / decision": "Monthly trend is explicitly qualified because dates are missing and March is sparse."},
                {"Approach": "Scatter / heatmap (alternatives)", "Strength / role": "Scatter plots expose relationships; heatmaps can scan state/channel/month combinations.", "Limitation / decision": "Scatter used for frequency vs revenue; a heatmap is deferred because only three sparse months are available."},
            ]
        )
        st.dataframe(alternatives, width="stretch", hide_index=True)
        st.markdown("**Tool and design choice:** pandas validates and aggregates structured workbook tables; scikit-learn compares reproducible baselines; Plotly provides hoverable, sortable visual evidence; Streamlit keeps the filters and management views accessible in one Python app. Bars, lines, distributions, and scatter plots are chosen for their data shape. The executive page prioritises five KPIs and concise findings; detailed technical, ethics, and governance content stays in supporting pages/expanders. Text labels supplement status colours, chart titles state the measure, and data tables provide exact values alongside hover visuals.")

    with st.expander("Ethics, privacy, fairness, bias, and ACS professional practice"):
        st.markdown(
            "**Privacy and minimisation:** customer names and email addresses are not displayed or used as model features. Customer-level exploration uses masked IDs; exports contain only the masked analytical feature table. Access to source-identifiable records should be restricted to authorised custodians.\n\n"
            "**Fairness and bias:** missing IDs/dates, uneven state coverage, excluded nonpositive sales, incomplete campaigns, and short historical coverage can distort comparisons. Do not infer sensitive traits or exclude groups automatically. Report missingness, use transparent thresholds, validate subgroup impacts, and require human review.\n\n"
            "**Deontology:** protect customer privacy and present limitations honestly, even when caveats complicate a business message. **Utilitarianism:** use controlled tests and human-reviewed alerts to seek customer/business benefit while limiting harm from bad forecasts or stock decisions. **Virtue ethics:** demonstrate honesty, fairness, care, competence, and transparency in analysis and communication.\n\n"
            "**ACS Code in practice:** Primacy of the Public Interest means privacy and fair treatment constrain targeting. Honesty means reporting p-values, source conflicts, missingness, and weak R² without overclaiming. Competence means validating joins and test performance before recommending action. Professionalism means documenting assumptions, restricting access, escalating data defects, and keeping management accountable for decisions."
        )

    with st.expander("Governance, accountability, and next data collection"):
        st.markdown(
            "**Proposed ownership (roles to be assigned by BlueCart):** retail operations owns sales definitions; CRM/customer-data stewardship owns customer identifiers and consent; marketing owns campaign metadata; inventory control owns SKU/stock records; analytics owns transformations, model versions, and dashboard release.\n\n"
            "**Controls:** role-based access to identifiable source data; scheduled missingness, duplicate-ID, numeric-range, and date-validity checks; versioned datasets, transformations, models, and threshold changes; recorded model assumptions and test metrics; drift/revalidation review before each release; auditable reason codes for each alert; and human approval before commercial action. Management remains accountable for resulting decisions.\n\n"
            "**Next data collection:** complete Order_Date and Customer_ID; Order_ID-to-SKU/Product_ID, category, quantity, unit price, and discount; campaign-to-order Campaign_ID and acquisition source with spend/conversions; customer tenure/consent state; inventory movements, lead time, and stock timestamps. These enable product demand, reliable inventory-sales analysis, measured campaign ROI, stronger customer value, and better validated prediction.\n\n"
            "**Change management:** log source extracts, cleaning changes, dashboard releases, model versions, metric definitions, and decision-rule threshold changes with owner, date, rationale, and approval."
        )

    with st.expander("Sprint 3 rubric coverage"):
        rubric = pd.DataFrame(
            [
                {"Criterion": "1. Executive summary", "Evidence": "Executive objective, priority KPIs, key findings, and concise next actions."},
                {"Criterion": "2. Visual and design quality", "Evidence": "Responsive layout, labelled KPIs, appropriate bar/line/scatter/distribution charts, and stated visual rationale."},
                {"Criterion": "3. Storytelling", "Evidence": "Business question connects validated data, pattern findings, caveats, decision rules, and recommendations."},
                {"Criterion": "4. Decision support", "Evidence": "Transparent customer, inventory, market, channel, and model thresholds with actions."},
                {"Criterion": "5. Alternatives", "Evidence": "Regression, tree, forest, boosting, RFM, K-Means, and visual-method comparison with limits."},
                {"Criterion": "6. Evaluation", "Evidence": "Tool, method, chart, audience rationale, and current model evaluation."},
                {"Criterion": "7. Ethics", "Evidence": "Privacy, fairness, bias sources/mitigations, ethical frameworks, and ACS conduct in practice."},
                {"Criterion": "8. Actions and risks", "Evidence": "Prioritised actions with evidence, expected impact, risk, owner, governance, and future data needs."},
            ]
        )
        st.dataframe(rubric, width="stretch", hide_index=True)
        st.caption("This checklist maps implemented evidence to each rubric criterion; the assessor determines the final grade.")

    with st.expander("Data-mining methods, Sprint 1 cleaning summary, and data dictionary"):
        st.markdown(
            "**Methods used:** descriptive pattern mining for state, monthly, customer, and channel performance; quartile-based RFM segmentation; IQR screening for unusual order values; and Sprint 2 regression as a limited predictive benchmark. Structured transactions, customer attributes, campaign engagement, and stock records are mined only where their keys and measures support a valid comparison."
        )
        st.markdown(
            "**Pattern families:** summaries and trends, customer segmentation, high/low-value and frequency groups, and outlier flags. Association rules, marketing ROI, product-level demand, and causal effects are unavailable because basket items, campaign spend/conversions, and reliable campaign/SKU-to-order links are absent. Privacy, missingness, short history, sample size, and changing customer behavior constrain interpretation."
        )
        st.subheader("Sprint 1 cleaning summary")
        st.dataframe(context["summary"].fillna("").astype(str), width="stretch", hide_index=True)
        st.subheader("Data dictionary")
        st.dataframe(context["dictionary"], width="stretch", hide_index=True)

def main() -> None:
    apply_sprint3_style()
    model = get_analytical_model()
    orders, customers, marketing, context = _prepare_views(model)
    model_results, linear_predictions = evaluate_sprint2_models(model["orders"])
    context["all_orders"] = model["orders"]
    context["all_customers"] = model["customer_features"]
    context["all_marketing"] = model["marketing"]
    context["marketing"] = marketing
    context["summary"] = model["summary"]
    context["dictionary"] = model["dictionary"]
    context["model_results"] = model_results
    context["linear_predictions"] = linear_predictions

    if context["date_mode"] == "Select range":
        st.session_state["sprint3_date_caption"] = "Selected date range excludes records with missing dates; customer features are recalculated in-scope."
    else:
        st.session_state["sprint3_date_caption"] = "All available orders are included; campaign and inventory tables remain independent operational views."

    page = context["page"]
    if page == "Executive Overview":
        _show_overview(orders, customers, context)
    elif page == "Customer Data Mining":
        _show_customer_mining(orders, customers, context["thresholds"], context["statistical_comparisons"])
    elif page == "Revenue & Trend Mining":
        _show_revenue_trends(orders, context)
    elif page == "Marketing & Inventory":
        _show_marketing_inventory(marketing, context["inventory"])
    elif page == "Ethics, Governance & Responsible Analytics":
        _show_ethics_governance(context["all_orders"], context["all_customers"], context["all_marketing"], context)
    else:
        _show_decision_support(orders, customers, marketing, context)
