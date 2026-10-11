from __future__ import annotations

from pathlib import Path

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

try:
    from build_sprint1_outputs import main as build_outputs
except ModuleNotFoundError:
    build_outputs = None


ROOT = Path(__file__).resolve().parent
CLEANED_WORKBOOK = ROOT / "Sprint1_Cleaned_Pack.xlsx"
CSV_OUTPUT = ROOT / "DataDriven_TeamName_Sprint1.csv"
REPORT_FILE = ROOT / "Sprint1_Report.md"

st.set_page_config(
    page_title="BlueCart Retail - Data Mining & Decision Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_workbook_data() -> dict[str, pd.DataFrame]:
    if (not CLEANED_WORKBOOK.exists() or not CSV_OUTPUT.exists()):
        if build_outputs is None:
            raise FileNotFoundError(
                "Required output files are missing and the build helper is unavailable. "
                "Please ensure Sprint1_Cleaned_Pack.xlsx and DataDriven_TeamName_Sprint1.csv exist."
            )
        build_outputs()

    sheets = {
        "summary": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Summary"),
        "transactions": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Transactions_Clean"),
        "customers": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Customers_Clean"),
        "marketing": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Marketing_Clean"),
        "inventory": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Inventory_Clean"),
        "analysis": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Analysis_Join"),
        "dictionary": pd.read_excel(CLEANED_WORKBOOK, sheet_name="Data_Dictionary"),
    }
    return sheets


def metric_card(label: str, value: object, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def apply_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy-900: #0B1645;
            --orange-500: #FF4B1F;
            --orange-600: #F63A10;
            --white: #F6F7FB;
            --line: rgba(255,255,255,0.10);
        }
        .stApp {
            background: linear-gradient(180deg, #0B1645 0%, #0A1238 100%);
            color: var(--white);
        }
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2.4rem;
            max-width: 1500px;
        }
        h1, h2, h3, h4, p {
            letter-spacing: -0.02em;
            color: var(--white);
        }
        .brand-strip {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0 0 1.2rem 0;
            padding: 0.05rem 0 0.5rem 0;
        }
        .brand-item {
            font-weight: 800;
            color: var(--orange-500);
            font-size: clamp(1.3rem, 2vw, 2.4rem);
            line-height: 1.1;
        }
        .brand-item.secondary {
            color: var(--white);
            font-weight: 700;
            font-size: clamp(1rem, 1.4vw, 1.6rem);
        }
        .assignment-shell {
            margin: 0.5rem 0 2rem 0;
        }
        .assignment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.2rem;
        }
        .assignment-title {
            font-size: clamp(2.3rem, 4vw, 4rem);
            font-weight: 900;
            letter-spacing: -0.04em;
            color: var(--orange-500);
            line-height: 1.1;
        }
        .brand-group {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.7rem;
            color: var(--white);
            font-weight: 800;
        }
        .brand-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 44px;
            height: 44px;
            border-radius: 12px;
            font-size: 1.7rem;
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--line);
        }
        .brand-label {
            font-size: clamp(1.1rem, 1.8vw, 2rem);
            color: var(--white);
        }
        .brand-label strong {
            font-weight: 900;
        }
        .assignment-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.4rem;
            margin-top: 1rem;
        }
        .assignment-card {
            background: linear-gradient(180deg, #ff5a2f 0%, #f14b1d 100%);
            border-radius: 18px;
            padding: 1.2rem 1.3rem 1.4rem 1.3rem;
            min-height: 340px;
            box-shadow: inset 0 -4px rgba(0,0,0,0.06);
        }
        .assignment-card h3 {
            margin: 0 0 0.8rem 0;
            font-size: clamp(2rem, 2.8vw, 3rem);
            line-height: 1.1;
            color: var(--white);
            font-weight: 900;
        }
        .assignment-card p,
        .assignment-card li {
            margin: 0;
            font-size: clamp(1.1rem, 1.55vw, 1.7rem);
            line-height: 1.5;
            color: rgba(255,255,255,0.98);
        }
        .assignment-card ul {
            padding-left: 1.1rem;
            margin: 0;
        }
        .assignment-card li + li {
            margin-top: 0.5rem;
        }
        .assignment-card .muted {
            color: rgba(255,255,255,0.92);
            font-weight: 600;
        }
        .sprint-shell {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.2rem 1.4rem 1.4rem 1.4rem;
            box-shadow: 0 18px 36px rgba(0,0,0,0.22);
            margin-bottom: 1.4rem;
        }
        .reference-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(250px, 1fr));
            gap: 1.2rem;
            margin-top: 1rem;
        }
        .reference-card {
            background: linear-gradient(180deg, #ff5b2e 0%, #f44b1d 100%);
            border-radius: 18px;
            padding: 1.2rem 1.1rem 1.3rem 1.1rem;
            min-height: 200px;
            box-shadow: inset 0 -4px rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
        }
        .reference-card.wide {
            grid-column: span 2;
        }
        .reference-card h3 {
            margin: 0.2rem 0 0.85rem 0;
            font-weight: 800;
            font-size: clamp(1.3rem, 2vw, 2.2rem);
            color: #fff;
        }
        .reference-card p {
            margin: 0;
            font-size: 1.06rem;
            line-height: 1.6;
            color: rgba(255,255,255,0.98);
        }
        .section-title {
            margin: 1.8rem 0 0.8rem 0;
            font-size: clamp(2rem, 2.6vw, 3rem);
            font-weight: 800;
            color: var(--orange-500);
        }
        .section-panel {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .stat-box {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            min-height: 110px;
        }
        .stat-box .label {
            display: block;
            color: rgba(255,255,255,0.74);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .stat-box .value {
            font-size: clamp(1.4rem, 2vw, 2rem);
            font-weight: 800;
            color: var(--white);
            line-height: 1.2;
        }
        .insight-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            min-height: 180px;
        }
        .insight-card strong {
            color: var(--orange-500);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .insight-card h4 {
            margin: 0.35rem 0 0.5rem 0;
            font-size: 1.15rem;
            color: var(--white);
        }
        .insight-card p {
            margin: 0.3rem 0 0 0;
            color: rgba(255,255,255,0.82);
            line-height: 1.55;
        }
        .model-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(220px, 1fr));
            gap: 1rem;
        }
        .model-card {
            background: linear-gradient(180deg, rgba(255, 75, 31, 0.12), rgba(255,255,255,0.03));
            border: 1px solid rgba(255, 75, 31, 0.35);
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }
        .model-card h4 {
            color: var(--orange-500);
            margin: 0 0 0.55rem 0;
            font-size: 1.1rem;
        }
        .model-card p {
            margin: 0;
            line-height: 1.55;
            color: rgba(255,255,255,0.96);
        }
        .placeholder {
            color: #ffe9df;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,0.18);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B1645 0%, #111f4d 100%);
        }
        section[data-testid="stSidebar"] * {
            color: var(--white) !important;
        }
        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }
        .stTabs [role="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.65rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    st.markdown(
        """
        <div class="assignment-shell">
            <div class="assignment-header">
                <div class="assignment-title">BlueCart</div>
                <div class="brand-group">
                    <div class="brand-icon">📈</div>
                    <div class="brand-label"><strong>Sales</strong> Performance</div>
                </div>
            </div>
            <div class="brand-strip" style="margin-top: 0.4rem; justify-content: flex-start;">
                <div class="brand-item" style="font-size: clamp(1.1rem, 1.8vw, 2.1rem);">Operational dashboard for revenue, channel, and inventory performance</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assignment_context() -> None:
    st.markdown(
        """
        <div class="sprint-shell">
            <div class="brand-item" style="font-size: 1.05rem; color: #ffffff; font-weight: 700;">Business overview</div>
            <p style="margin-top: 0.7rem; line-height: 1.6; color: rgba(255,255,255,0.9);">
                Monitor sales performance, compare regional revenue contribution, evaluate marketing efficiency, and identify inventory risk areas using a single operational view.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_section_title(title: str) -> None:
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)


def clean_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Order_Date"] = pd.to_datetime(result["Order_Date"], errors="coerce")
    result["Total_Sales"] = pd.to_numeric(result["Total_Sales"], errors="coerce")
    result["State"] = result["State"].fillna("Unassigned")
    result["State"] = result["State"].replace({"Ca": "California", "Or": "Oregon"})
    return result


def clean_marketing_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Campaign_Date"] = pd.to_datetime(result["Campaign_Date"], errors="coerce")
    result["Reach"] = pd.to_numeric(result["Reach"], errors="coerce")
    result["CTR"] = pd.to_numeric(result["CTR"], errors="coerce")
    return result


def clean_inventory_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Stock"] = pd.to_numeric(result["Stock"], errors="coerce")
    result["Unit_Price_Amount"] = pd.to_numeric(result["Unit_Price_Amount"], errors="coerce")
    return result


def build_baseline_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    model_df = df.copy()
    model_df = model_df.dropna(subset=["Total_Sales", "Order_Date", "State"]).copy()
    model_df["Order_Date"] = pd.to_datetime(model_df["Order_Date"], errors="coerce")
    model_df["Month"] = model_df["Order_Date"].dt.month
    model_df["DayOfMonth"] = model_df["Order_Date"].dt.day
    model_df["Year"] = model_df["Order_Date"].dt.year

    features = ["State", "Month", "DayOfMonth", "Year"]
    X = model_df[features]
    y = model_df["Total_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("state", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["State"]),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["Prior_Purchases", "Month", "Day_of_Week"]),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, prediction)),
        "MSE": float(mean_squared_error(y_test, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "R2": float(r2_score(y_test, prediction)),
    }

    summary = pd.DataFrame(
        {
            "Actual": y_test.reset_index(drop=True),
            "Predicted": prediction,
            "Error": prediction - y_test.reset_index(drop=True),
        }
    )
    return model, metrics, summary


def sprint2_legacy_main() -> None:
    apply_dashboard_style()
    render_brand_header()
    render_assignment_context()
    data = load_workbook_data()

    analysis = clean_analysis_frame(data["analysis"])
    marketing = clean_marketing_frame(data["marketing"])
    inventory = clean_inventory_frame(data["inventory"])
    customers = data["customers"].copy()
    summary = data["summary"].copy()
    dictionary = data["dictionary"].copy()

    sales_only = analysis.dropna(subset=["Total_Sales"])
    date_values = sales_only.dropna(subset=["Order_Date"])
    min_date = date_values["Order_Date"].min()
    max_date = date_values["Order_Date"].max()
    date_range = (min_date, max_date) if pd.notna(min_date) and pd.notna(max_date) else None

    with st.sidebar:
        st.markdown("## Dashboard filters")
        st.caption("Adjust the operating view by region, channel, and date range.")
        state_options = ["All"] + sorted([state for state in analysis["State"].dropna().unique() if state != "Unassigned"])
        channel_options = ["All"] + sorted(marketing["Channel"].dropna().unique().tolist())
        selected_state = st.selectbox("State", state_options)
        selected_channel = st.selectbox("Marketing channel", channel_options)
        if date_range:
            selected_dates = st.date_input("Order date range", value=(date_range[0].date(), date_range[1].date()))
        else:
            selected_dates = None
        show_joined_only = st.checkbox("Joined orders only", value=False)
        st.divider()
        st.markdown("### Exports")
        st.download_button("Download cleaned CSV", data=CSV_OUTPUT.read_bytes(), file_name=CSV_OUTPUT.name, mime="text/csv", width="stretch")
        st.download_button("Download cleaned workbook", data=CLEANED_WORKBOOK.read_bytes(), file_name=CLEANED_WORKBOOK.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

    if selected_dates and len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])
    else:
        start_date = None
        end_date = None

    filtered_analysis = analysis.copy()
    if selected_state != "All":
        filtered_analysis = filtered_analysis[filtered_analysis["State"] == selected_state]
    if start_date is not None and end_date is not None:
        filtered_analysis = filtered_analysis[filtered_analysis["Order_Date"].between(start_date, end_date, inclusive="both")]
    if show_joined_only:
        filtered_analysis = filtered_analysis[filtered_analysis["Customer_Name"].notna()]

    filtered_marketing = marketing.copy()
    if selected_channel != "All":
        filtered_marketing = filtered_marketing[filtered_marketing["Channel"] == selected_channel]

    identifiable_sales = filtered_analysis.dropna(subset=["Customer_ID_Masked", "Total_Sales"]).copy()
    customer_frequency = identifiable_sales.groupby("Customer_ID_Masked")["Order_ID"].transform("count")
    identifiable_sales["Customer_Type"] = np.where(customer_frequency > 1, "Repeat customer", "One-time customer")
    customer_type_sales = (
        identifiable_sales.groupby("Customer_Type", as_index=False)["Total_Sales"]
        .agg(Transactions="count", Average_Sales="mean")
    )

    total_sales = float(filtered_analysis["Total_Sales"].sum(skipna=True)) if not filtered_analysis.empty else 0.0
    avg_sales = float(filtered_analysis["Total_Sales"].mean(skipna=True)) if not filtered_analysis.empty else 0.0
    order_count = int(filtered_analysis["Order_ID"].nunique()) if not filtered_analysis.empty else 0
    customer_count = int(filtered_analysis["Customer_ID_Masked"].nunique(dropna=True)) if not filtered_analysis.empty else 0
    joined_count = int(filtered_analysis["Customer_Name"].notna().sum()) if not filtered_analysis.empty else 0
    avg_ctr = float(filtered_marketing["CTR"].mean(skipna=True)) if not filtered_marketing.empty else 0.0
    overall_sales = analysis["Total_Sales"].dropna()
    overall_avg_sales = float(overall_sales.mean()) if not overall_sales.empty else 0.0

    state_sales = (
        filtered_analysis.dropna(subset=["Total_Sales"])
        .groupby("State", as_index=False)["Total_Sales"]
        .sum()
        .sort_values("Total_Sales", ascending=False)
    ) if not filtered_analysis.empty else pd.DataFrame(columns=["State", "Total_Sales"])

    channel_ctr = (
        filtered_marketing.dropna(subset=["CTR"])
        .groupby("Channel", as_index=False)["CTR"]
        .mean()
        .sort_values("CTR", ascending=False)
    ) if not filtered_marketing.empty else pd.DataFrame(columns=["Channel", "CTR"])

    inventory_summary = inventory.dropna(subset=["Stock"]).sort_values("Stock", ascending=True)
    top_state = state_sales.iloc[0]["State"] if not state_sales.empty else "[Insert top state]"
    top_state_sales = state_sales.iloc[0]["Total_Sales"] if not state_sales.empty else "[Insert total sales]"
    top_channel = channel_ctr.iloc[0]["Channel"] if not channel_ctr.empty else "[Insert channel]"
    top_channel_ctr = channel_ctr.iloc[0]["CTR"] if not channel_ctr.empty else "[Insert CTR]"
    lowest_stock_product = inventory_summary.iloc[0]["Product_Name"] if not inventory_summary.empty else "[Insert product]"

    display_section_title("Executive Summary")
    eda_cols = st.columns(4)
    with eda_cols[0]:
        st.markdown("<div class='stat-box'><span class='label'>Dataset rows</span><div class='value'>%s</div></div>" % (len(analysis)), unsafe_allow_html=True)
    with eda_cols[1]:
        st.markdown("<div class='stat-box'><span class='label'>Observed sales</span><div class='value'>%s</div></div>" % (int(overall_sales.shape[0])), unsafe_allow_html=True)
    with eda_cols[2]:
        st.markdown("<div class='stat-box'><span class='label'>Average order value</span><div class='value'>%s</div></div>" % (f"${overall_avg_sales:,.2f}"), unsafe_allow_html=True)
    with eda_cols[3]:
        st.markdown("<div class='stat-box'><span class='label'>Average marketing CTR</span><div class='value'>%s</div></div>" % (f"{avg_ctr:.2%}"), unsafe_allow_html=True)

    display_section_title("Business Questions")
    st.markdown(
        """
        Use the dashboard to compare revenue by state, understand repeat-customer behaviour, monitor the available sales trend, review channel engagement, and identify inventory pressure.
        """
    )

    display_section_title("Performance Overview")
    viz_cols = st.columns(2)
    with viz_cols[0]:
        st.subheader("Sales concentration by state")
        if not state_sales.empty:
            fig_state = px.bar(state_sales, x="State", y="Total_Sales", color="Total_Sales", color_continuous_scale=["#fef3c7", "#f59e0b", "#b45309"], title="Sales concentration by state")
            fig_state.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_state, width="stretch")
            st.caption("Largest revenue contribution by state in the current selection.")
        else:
            st.info("No sales data available for the selected filters.")
    with viz_cols[1]:
        st.subheader("Daily sales trend")
        if not filtered_analysis.empty:
            daily_sales = (
                filtered_analysis.dropna(subset=["Order_Date", "Total_Sales"])
                .assign(Order_Date=lambda df: df["Order_Date"].dt.date)
                .groupby("Order_Date", as_index=False)["Total_Sales"]
                .sum()
                .sort_values("Order_Date")
            )
            if not daily_sales.empty:
                fig_time = px.line(daily_sales, x="Order_Date", y="Total_Sales", markers=True, title="Daily sales trend")
                fig_time.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_time, width="stretch")
                st.caption("Daily sales movement across the selected period.")
            else:
                st.info("No dated sales data available for the selected filters.")
        else:
            st.info("No sales data available for the selected filters.")

    viz_cols_2 = st.columns(2)
    with viz_cols_2[0]:
        st.subheader("Marketing engagement by channel")
        if not channel_ctr.empty:
            fig_ctr = px.bar(channel_ctr, x="Channel", y="CTR", color="CTR", color_continuous_scale=["#f0f9ff", "#60a5fa", "#1d4ed8"], title="Average CTR by channel")
            fig_ctr.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_ctr, width="stretch")
            st.caption("Channel engagement performance based on average CTR.")
        else:
            st.info("No marketing data available for the selected filters.")
    with viz_cols_2[1]:
        st.subheader("Average value by customer type")
        if not customer_type_sales.empty:
            fig_customer_type = px.bar(customer_type_sales, x="Customer_Type", y="Average_Sales", color="Customer_Type", title="Average transaction value by customer type", color_discrete_sequence=["#38bdf8", "#f59e0b"])
            fig_customer_type.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Average sales")
            st.plotly_chart(fig_customer_type, width="stretch")
            st.caption("Comparison of average observed sales for repeat and one-time customers.")
        else:
            st.info("No identifiable customer sales available for the selected filters.")

    display_section_title("Inventory Monitoring")
    if not inventory_summary.empty:
        fig_stock = px.bar(inventory_summary.head(12), x="Product_Name", y="Stock", color="Stock", color_continuous_scale=["#fef2f2", "#f87171", "#991b1b"], title="Lowest-stock products")
        fig_stock.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Product")
        st.plotly_chart(fig_stock, width="stretch")
        st.caption("Products with the lowest stock levels and highest replenishment urgency.")
    else:
        st.info("No inventory data available.")

    model_df = analysis.dropna(subset=["Total_Sales"]).copy()
    model_df["Order_Date"] = pd.to_datetime(model_df["Order_Date"], errors="coerce")
    model_df = model_df.sort_values(["Customer_ID_Masked", "Order_Date", "Order_ID"], na_position="last")
    model_df["Prior_Purchases"] = model_df.groupby("Customer_ID_Masked")["Order_ID"].cumcount()
    model_df["Month"] = model_df["Order_Date"].dt.month
    model_df["Day_of_Week"] = model_df["Order_Date"].dt.dayofweek
    model_df["Month"] = model_df["Month"].fillna(0)
    model_df["Day_of_Week"] = model_df["Day_of_Week"].fillna(-1)

    feature_cols = ["State", "Prior_Purchases", "Month", "Day_of_Week"]
    X = model_df[feature_cols]
    y = model_df["Total_Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("state", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["State"]),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["Prior_Purchases", "Month", "Day_of_Week"]),
        ],
        remainder="drop",
    )

    model_specs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    model_rows = []
    for name, estimator in model_specs.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = {
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "R2": r2_score(y_test, preds),
        }
        model_rows.append(metrics)

    comparison_df = pd.DataFrame(model_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_model = comparison_df.iloc[0].to_dict()
    selected_model_name = best_model["Model"]
    selected_model_row = comparison_df[comparison_df["Model"] == selected_model_name].iloc[0]

    selected_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model_specs[selected_model_name])])
    selected_pipeline.fit(X_train, y_train)
    selected_preds = selected_pipeline.predict(X_test)
    prediction_df = pd.DataFrame(
        {
            "Actual": y_test.reset_index(drop=True),
            "Predicted": selected_preds,
            "Error": selected_preds - y_test.reset_index(drop=True),
        }
    )

    home_tab, dashboard_tab = st.tabs(["Home", "Dashboard"])

    with home_tab:
        st.subheader("Selected baseline model")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Best model", selected_model_name)
        metric_cols[1].metric("MAE", f"${selected_model_row['MAE']:,.2f}")
        metric_cols[2].metric("RMSE", f"${selected_model_row['RMSE']:,.2f}")
        metric_cols[3].metric("R²", f"{selected_model_row['R2']:.3f}")

        st.caption("Best-performing baseline model selected using the lowest RMSE on the validation split.")

        selected_scatter = px.scatter(
            prediction_df,
            x="Actual",
            y="Predicted",
            title=f"Actual vs predicted sales for {selected_model_name}",
            color_discrete_sequence=["#f59e0b"],
        )
        selected_scatter.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(selected_scatter, width="stretch")

        residual_hist = px.histogram(
            prediction_df,
            x="Error",
            nbins=20,
            title="Residual distribution for the selected model",
            color_discrete_sequence=["#ef4444"],
        )
        residual_hist.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(residual_hist, width="stretch")

    with dashboard_tab:
        display_section_title("Baseline model validation")
        st.markdown(
            """
            Each model was evaluated on the same train/test split to maintain a consistent benchmark. The comparison focuses on forecast error, with RMSE used as the primary selection metric.
            """
        )

        st.subheader("Model performance summary")
        st.dataframe(comparison_df.round(4), width="stretch", hide_index=True)

        comparison_chart = px.bar(
            comparison_df,
            x="Model",
            y="RMSE",
            color="Model",
            title="Model comparison by RMSE",
            color_discrete_sequence=["#f59e0b", "#f97316", "#fb7185"],
        )
        comparison_chart.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(comparison_chart, width="stretch")

        st.subheader("Model selection & justification")
        st.markdown(
            f"""
            **{selected_model_name}** was selected as the baseline forecast model because it delivered the lowest RMSE across the tested options. It provides the most practical starting point for operational forecasting while remaining easy to interpret.
            """
        )

        st.subheader("Prediction quality and residual check")
        prediction_preview = prediction_df.head(10).copy()
        prediction_preview["Actual"] = prediction_preview["Actual"].map(lambda x: f"${x:,.2f}")
        prediction_preview["Predicted"] = prediction_preview["Predicted"].map(lambda x: f"${x:,.2f}")
        prediction_preview["Error"] = prediction_preview["Error"].map(lambda x: f"${x:,.2f}")
        st.dataframe(prediction_preview, width="stretch", hide_index=True)

        actual_vs_pred = px.scatter(
            prediction_df,
            x="Actual",
            y="Predicted",
            title="Actual versus predicted sales",
            color_discrete_sequence=["#0ea5e9"],
        )
        actual_vs_pred.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(actual_vs_pred, width="stretch")

        residual_hist = px.histogram(
            prediction_df,
            x="Error",
            nbins=20,
            title="Residual histogram for the selected model",
            color_discrete_sequence=["#8b5cf6"],
        )
        residual_hist.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(residual_hist, width="stretch")

        st.info("Operational notes: use these views to prioritise revenue focus, campaign efficiency, and stock risk monitoring.")

        display_section_title("Operational Risks")
        st.markdown(
            """
            - Data quality remains a constraint where state labels and missing values can distort regional comparisons.
            - The current model is a baseline and should not drive automated commercial decisions without additional validation.
            - Inventory and channel patterns should be reviewed alongside profit, spend, and conversion data before actioning plans.
            """
        )

        display_section_title("Recommendations")
        st.markdown(
            """
            - Prioritise regional reviews in the highest-contributing states.
            - Use CTR to guide channel testing, but validate with conversion and spend data before scaling investment.
            - Review low-stock items and adjust replenishment plans for operational continuity.
            - Expand the forecast with richer sales, product, and promotional features before production use.
            """
        )

        tab_overview, tab_cleaning, tab_customer, tab_marketing, tab_inventory, tab_dictionary = st.tabs(["Overview", "Cleaning", "Customer view", "Marketing view", "Inventory view", "Data dictionary"])

        with tab_overview:
            left, right = st.columns([1.1, 0.9], gap="large")
            with left:
                st.subheader("Sales by state")
                if not state_sales.empty:
                    fig_state = px.bar(state_sales, x="State", y="Total_Sales", color="Total_Sales", color_continuous_scale=["#fef3c7", "#f59e0b", "#b45309"], title="Sales concentration by state")
                    fig_state.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_state, width="stretch")
                else:
                    st.info("No sales rows match the current filters.")
            with right:
                st.subheader("Sales over time")
                daily_sales = (
                    filtered_analysis.dropna(subset=["Order_Date", "Total_Sales"])
                    .assign(Order_Date=lambda frame: frame["Order_Date"].dt.date)
                    .groupby("Order_Date", as_index=False)["Total_Sales"]
                    .sum()
                    .sort_values("Order_Date")
                ) if not filtered_analysis.empty else pd.DataFrame(columns=["Order_Date", "Total_Sales"])
                if not daily_sales.empty:
                    fig_time = px.line(daily_sales, x="Order_Date", y="Total_Sales", markers=True, title="Daily sales trend")
                    fig_time.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_time, width="stretch")
                else:
                    st.info("No dated sales rows match the current filters.")

        with tab_cleaning:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Transactions rows", int(data["transactions"].shape[0]))
            with c2:
                st.metric("Customers rows", int(customers.shape[0]))
            with c3:
                st.metric("Marketing rows", int(marketing.shape[0]))
            with c4:
                st.metric("Inventory rows", int(inventory.shape[0]))
            st.markdown("### Cleaning summary")
            st.dataframe(summary.fillna("").astype(str), width="stretch", hide_index=True)

        with tab_customer:
            c1, c2 = st.columns([1.05, 0.95], gap="large")
            with c1:
                st.subheader("Customer distribution by state")
                customer_states = customers.copy()
                if "State" in customer_states.columns:
                    customer_states["State"] = customer_states["State"].fillna("Unknown")
                    customer_state_counts = customer_states.groupby("State", as_index=False).size().sort_values("size", ascending=False)
                    fig_customer_state = px.bar(customer_state_counts, x="State", y="size", title="Customer counts by state", color="size", color_continuous_scale=["#e0f2fe", "#38bdf8", "#0f766e"])
                    fig_customer_state.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_customer_state, width="stretch")
            with c2:
                st.subheader("Privacy masking check")
                privacy_preview = customers[[col for col in customers.columns if col in ["Customer_ID_Masked", "Email_Masked", "State", "Customer_Name"]]]
                st.dataframe(privacy_preview.head(12), width="stretch", hide_index=True)

        with tab_marketing:
            c1, c2 = st.columns([1.05, 0.95], gap="large")
            with c1:
                st.subheader("CTR by channel")
                if not channel_ctr.empty:
                    fig_ctr = px.bar(channel_ctr, x="Channel", y="CTR", title="Average click-through rate by channel", color="CTR", color_continuous_scale=["#f0f9ff", "#60a5fa", "#1d4ed8"])
                    fig_ctr.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_ctr, width="stretch")
                else:
                    st.info("No CTR values available for the selected filter.")
            with c2:
                st.subheader("Reach vs CTR")
                scatter_data = filtered_marketing.dropna(subset=["Reach", "CTR"])
                if not scatter_data.empty:
                    fig_scatter = px.scatter(scatter_data, x="Reach", y="CTR", color="Channel", title="Marketing reach and engagement")
                    fig_scatter.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_scatter, width="stretch")
                else:
                    st.info("No marketing rows match the current filter.")

        with tab_inventory:
            c1, c2 = st.columns([1.05, 0.95], gap="large")
            with c1:
                st.subheader("Stock by product")
                stock_plot = inventory.sort_values("Stock", ascending=True).head(20)
                fig_stock = px.bar(stock_plot, x="Product_Name", y="Stock", color="Stock", color_continuous_scale=["#fef2f2", "#f87171", "#991b1b"], title="Lowest-stock products")
                fig_stock.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Product")
                st.plotly_chart(fig_stock, width="stretch")
            with c2:
                st.subheader("Price distribution")
                price_data = inventory.dropna(subset=["Unit_Price_Amount"])
                fig_price = px.histogram(price_data, x="Unit_Price_Amount", nbins=12, title="Unit price distribution", color_discrete_sequence=["#f59e0b"])
                fig_price.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_price, width="stretch")

        with tab_dictionary:
            st.subheader("Data dictionary")
            st.dataframe(dictionary, width="stretch", hide_index=True)


if __name__ == "__main__":
    import importlib
    import sprint3_analytics
    import sprint3_dashboard

    importlib.reload(sprint3_analytics)
    importlib.reload(sprint3_dashboard)
    sprint3_dashboard.main()
