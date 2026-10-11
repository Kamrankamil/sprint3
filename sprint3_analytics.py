from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "Sprint1_Cleaned_Pack.xlsx"


SHEETS = {
    "transactions": "Transactions_Clean",
    "customers": "Customers_Clean",
    "marketing": "Marketing_Clean",
    "inventory": "Inventory_Clean",
    "summary": "Summary",
    "dictionary": "Data_Dictionary",
}
RAW_WORKBOOK = ROOT / "BlueCart_DataVault_Sprint1_Dataset (1).xlsx"
ALTERNATE_CSV = ROOT / "DataDriven_BlueCart_Sprint1 (1).csv"


def _clean_identifier(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip().str.upper()
    return cleaned.mask(cleaned.eq(""))


def _clean_state(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip().str.title()
    cleaned = cleaned.replace({"Ca": "California", "Or": "Oregon", "Wa": "Washington"})
    return cleaned.mask(cleaned.eq(""))


def _clean_channel(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return cleaned.mask(cleaned.eq(""))


def load_source_tables(workbook: Path = WORKBOOK) -> dict[str, pd.DataFrame]:
    """Load the existing cleaned tables without changing the source workbook."""
    return {name: pd.read_excel(workbook, sheet_name=sheet) for name, sheet in SHEETS.items()}


def standardize_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    transactions = tables["transactions"].copy()
    customers = tables["customers"].copy()
    marketing = tables["marketing"].copy()
    inventory = tables["inventory"].copy()

    transactions["Order_ID"] = _clean_identifier(transactions["Order_ID"])
    transactions["Customer_ID_Masked"] = _clean_identifier(transactions["Customer_ID_Masked"])
    transactions["Order_Date"] = pd.to_datetime(transactions["Order_Date"], errors="coerce")
    transactions["Total_Sales"] = pd.to_numeric(transactions["Total_Sales"], errors="coerce")

    customers["Customer_ID_Masked"] = _clean_identifier(customers["Customer_ID_Masked"])
    customers["State"] = _clean_state(customers["State"])

    marketing["Campaign_ID"] = _clean_identifier(marketing["Campaign_ID"])
    marketing["Campaign_Date"] = pd.to_datetime(marketing["Campaign_Date"], errors="coerce")
    marketing["Reach"] = pd.to_numeric(marketing["Reach"], errors="coerce")
    marketing["CTR"] = pd.to_numeric(marketing["CTR"], errors="coerce")
    marketing["Channel"] = _clean_channel(marketing["Channel"])

    inventory["SKU"] = _clean_identifier(inventory["SKU"])
    inventory["Stock"] = pd.to_numeric(inventory["Stock"], errors="coerce")
    inventory["Unit_Price_Amount"] = pd.to_numeric(inventory["Unit_Price_Amount"], errors="coerce")

    return {
        **tables,
        "transactions": transactions,
        "customers": customers,
        "marketing": marketing,
        "inventory": inventory,
    }


def _value_band(value: float, lower: float, upper: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value <= lower:
        return "Low"
    if value <= upper:
        return "Typical"
    return "High"


def add_order_features(transactions: pd.DataFrame) -> pd.DataFrame:
    orders = transactions.copy()
    orders["Year"] = orders["Order_Date"].dt.year.astype("Int64")
    orders["Month"] = orders["Order_Date"].dt.month.astype("Int64")
    orders["Quarter"] = orders["Order_Date"].dt.quarter.astype("Int64")
    orders["Day_of_Week"] = orders["Order_Date"].dt.dayofweek.astype("Int64")
    orders["Day_of_Week_Name"] = orders["Order_Date"].dt.day_name()

    observed_sales = orders["Total_Sales"].dropna()
    if observed_sales.empty:
        orders["Order_Value_Band"] = "Unknown"
    else:
        lower = float(observed_sales.quantile(0.25))
        upper = float(observed_sales.quantile(0.75))
        orders["Order_Value_Band"] = orders["Total_Sales"].map(
            lambda value: _value_band(value, lower, upper)
        )
    return orders


def _rank_quartile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    rank = values.rank(method="average", pct=True)
    score = np.ceil(rank * 4).clip(1, 4)
    if not higher_is_better:
        score = 5 - score
    return score.astype("Int64")


def build_customer_features(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | pd.Timestamp | None]]:
    identified = orders.dropna(subset=["Customer_ID_Masked"])
    if identified.empty:
        return pd.DataFrame(), {"revenue_p25": 0.0, "revenue_p75": 0.0, "frequency_p25": 0.0, "frequency_p75": 0.0, "recency_p25": 0.0, "recency_p75": 0.0, "reference_date": None}

    customers = identified.groupby("Customer_ID_Masked", as_index=False).agg(
        State=("State", "first"),
    )
    observed_sales = identified.dropna(subset=["Total_Sales"])
    observed_customer_features = (
        observed_sales.groupby("Customer_ID_Masked", as_index=False)
        .agg(
            Customer_Purchase_Count=("Order_ID", "nunique"),
            Customer_Total_Revenue=("Total_Sales", lambda values: values.sum(min_count=1)),
            Customer_Average_Order_Value=("Total_Sales", "mean"),
            Customer_Last_Purchase=("Order_Date", "max"),
        )
    )
    customers = customers.merge(
        observed_customer_features,
        on="Customer_ID_Masked",
        how="left",
        validate="one_to_one",
    )
    customers["Customer_Purchase_Count"] = customers["Customer_Purchase_Count"].fillna(0).astype(int)
    reference_date = observed_sales["Order_Date"].max()
    customers["Customer_Frequency"] = customers["Customer_Purchase_Count"]
    customers["Customer_Monetary_Value"] = customers["Customer_Total_Revenue"]
    customers["Customer_Recency"] = (reference_date - customers["Customer_Last_Purchase"]).dt.days
    customers["Customer_Type"] = np.select(
        [
            customers["Customer_Purchase_Count"].gt(1),
            customers["Customer_Purchase_Count"].eq(1),
        ],
        ["Repeat", "One-Time"],
        default="Insufficient Sales Data",
    )

    revenue = customers["Customer_Total_Revenue"].dropna()
    dated_recency = customers["Customer_Recency"].dropna()
    repeat_recency = customers.loc[
        customers["Customer_Purchase_Count"] > 1, "Customer_Recency"
    ].dropna()
    revenue_p25 = float(revenue.quantile(0.25)) if not revenue.empty else 0.0
    revenue_p75 = float(revenue.quantile(0.75)) if not revenue.empty else 0.0
    observed_frequency = customers.loc[
        customers["Customer_Purchase_Count"] > 0, "Customer_Frequency"
    ]
    frequency_p25 = float(observed_frequency.quantile(0.25)) if not observed_frequency.empty else 0.0
    frequency_p75 = float(observed_frequency.quantile(0.75)) if not observed_frequency.empty else 0.0
    customers["Customer_Frequency_Band"] = np.select(
        [
            customers["Customer_Purchase_Count"].eq(0),
            customers["Customer_Frequency"].le(frequency_p25),
            customers["Customer_Frequency"].gt(frequency_p75),
        ],
        ["Insufficient Sales Data", "Low (<=P25)", "High (>P75)"],
        default="Typical",
    )
    recency_p25 = float(dated_recency.quantile(0.25)) if not dated_recency.empty else 0.0
    recency_p75 = float(repeat_recency.quantile(0.75)) if not repeat_recency.empty else 0.0

    def assign_segment(row: pd.Series) -> str:
        if row["Customer_Purchase_Count"] == 0 or pd.isna(row["Customer_Total_Revenue"]):
            return "Insufficient Sales Data"
        if row["Customer_Total_Revenue"] > revenue_p75:
            return "High Value"
        if (
            row["Customer_Purchase_Count"] > 1
            and pd.notna(row["Customer_Recency"])
            and row["Customer_Recency"] >= recency_p75
        ):
            return "Recency Review (Limited Data)"
        if row["Customer_Purchase_Count"] > 1 and row["Customer_Frequency"] >= frequency_p75:
            return "Loyal / High Frequency"
        if row["Customer_Purchase_Count"] == 1:
            return "One-Time"
        if row["Customer_Total_Revenue"] <= revenue_p25:
            return "Low Value"
        if pd.notna(row["Customer_Recency"]) and row["Customer_Recency"] <= recency_p25:
            return "Recent"
        return "Developing"

    customers["Customer_Segment"] = customers.apply(assign_segment, axis=1)
    customers["R_Score"] = _rank_quartile(customers["Customer_Recency"], higher_is_better=False)
    customers["F_Score"] = _rank_quartile(
        customers["Customer_Frequency"].where(customers["Customer_Purchase_Count"] > 0)
    )
    customers["M_Score"] = _rank_quartile(customers["Customer_Monetary_Value"])
    customers["RFM_Score"] = (
        customers[["R_Score", "F_Score", "M_Score"]]
        .astype("string")
        .fillna("")
        .agg("".join, axis=1)
    )
    customers.loc[customers[["R_Score", "F_Score", "M_Score"]].isna().any(axis=1), "RFM_Score"] = pd.NA

    thresholds: dict[str, float | pd.Timestamp | None] = {
        "revenue_p25": revenue_p25,
        "revenue_p75": revenue_p75,
        "frequency_p25": frequency_p25,
        "frequency_p75": frequency_p75,
        "recency_p25": recency_p25,
        "recency_p75": recency_p75,
        "reference_date": reference_date if pd.notna(reference_date) else None,
    }
    return customers, thresholds


def compare_customer_and_state_means(orders: pd.DataFrame) -> dict[str, Any]:
    """Return Welch and one-way ANOVA comparisons for observed positive sales."""
    observed = orders.dropna(subset=["Total_Sales"]).copy()
    state_groups = [
        group["Total_Sales"].to_numpy()
        for _, group in observed.loc[observed["State"] != "Unassigned"].groupby("State")
        if len(group) >= 2
    ]
    if len(state_groups) >= 2:
        state_test = f_oneway(*state_groups)
        state_result: dict[str, Any] = {
            "statistic": float(state_test.statistic),
            "p_value": float(state_test.pvalue),
            "groups": len(state_groups),
            "n": int(sum(map(len, state_groups))),
        }
    else:
        state_result = {"statistic": None, "p_value": None, "groups": len(state_groups), "n": int(sum(map(len, state_groups)))}

    customer_counts = (
        observed.dropna(subset=["Customer_ID_Masked"])
        .groupby("Customer_ID_Masked")["Order_ID"]
        .nunique()
    )
    observed["Customer_Type"] = observed["Customer_ID_Masked"].map(
        lambda customer_id: "Unidentified" if pd.isna(customer_id) else (
            "Repeat" if customer_counts.get(customer_id, 0) > 1 else "One-Time"
        )
    )
    repeat_values = observed.loc[observed["Customer_Type"] == "Repeat", "Total_Sales"]
    one_time_values = observed.loc[observed["Customer_Type"] == "One-Time", "Total_Sales"]
    if len(repeat_values) >= 2 and len(one_time_values) >= 2:
        customer_test = ttest_ind(repeat_values, one_time_values, equal_var=False)
        customer_result: dict[str, Any] = {
            "statistic": float(customer_test.statistic),
            "p_value": float(customer_test.pvalue),
            "repeat_n": int(len(repeat_values)),
            "repeat_mean": float(repeat_values.mean()),
            "one_time_n": int(len(one_time_values)),
            "one_time_mean": float(one_time_values.mean()),
        }
    else:
        customer_result = {
            "statistic": None,
            "p_value": None,
            "repeat_n": int(len(repeat_values)),
            "repeat_mean": float(repeat_values.mean()) if len(repeat_values) else None,
            "one_time_n": int(len(one_time_values)),
            "one_time_mean": float(one_time_values.mean()) if len(one_time_values) else None,
        }
    return {"state_anova": state_result, "customer_welch": customer_result}


def audit_source_reconciliation(transactions: pd.DataFrame) -> dict[str, Any]:
    """Compare cleaned transaction facts against the raw workbook and alternate CSV."""
    result: dict[str, Any] = {
        "raw_workbook_available": RAW_WORKBOOK.exists(),
        "alternate_csv_available": ALTERNATE_CSV.exists(),
        "raw_rows": None,
        "matched_order_ids": None,
        "nonpositive_sales_removed": None,
        "clean_sales_disagree_with_raw_positive": None,
        "alternate_csv_sales_disagreements": None,
        "alternate_csv_positive_for_raw_nonpositive": None,
    }
    clean = transactions[["Order_ID", "Total_Sales"]].copy()
    clean["Order_ID"] = _clean_identifier(clean["Order_ID"])

    if RAW_WORKBOOK.exists():
        raw = pd.read_excel(RAW_WORKBOOK, sheet_name="Transactions_Raw")
        raw["Order_ID"] = _clean_identifier(raw["Order_ID"])
        raw["Total_Sales"] = pd.to_numeric(raw["Total_Sales"], errors="coerce")
        joined = raw[["Order_ID", "Total_Sales"]].merge(
            clean,
            on="Order_ID",
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_raw", "_clean"),
        )
        result["raw_rows"] = int(len(raw))
        result["matched_order_ids"] = int(joined["_merge"].eq("both").sum())
        result["nonpositive_sales_removed"] = int(raw["Total_Sales"].le(0).sum())
        raw_positive = joined["Total_Sales_raw"].gt(0)
        result["clean_sales_disagree_with_raw_positive"] = int(
            (raw_positive & joined["Total_Sales_clean"].ne(joined["Total_Sales_raw"])).sum()
        )

    if ALTERNATE_CSV.exists():
        alternate = pd.read_csv(ALTERNATE_CSV)
        alternate["Order_ID"] = _clean_identifier(alternate["Order_ID"])
        alternate["Total_Sales"] = pd.to_numeric(alternate["Total_Sales"], errors="coerce")
        if "Customer_ID_Masked" in alternate.columns:
            alternate["Customer_ID_Masked"] = _clean_identifier(alternate["Customer_ID_Masked"])
        raw_for_compare = (
            pd.read_excel(RAW_WORKBOOK, sheet_name="Transactions_Raw")
            if RAW_WORKBOOK.exists()
            else pd.DataFrame(columns=["Order_ID", "Total_Sales"])
        )
        if not raw_for_compare.empty:
            raw_for_compare["Order_ID"] = _clean_identifier(raw_for_compare["Order_ID"])
            raw_for_compare["Total_Sales"] = pd.to_numeric(raw_for_compare["Total_Sales"], errors="coerce")
            alternate_join = raw_for_compare[["Order_ID", "Total_Sales"]].merge(
                alternate[["Order_ID", "Total_Sales"]],
                on="Order_ID",
                how="inner",
                validate="one_to_one",
                suffixes=("_raw", "_alternate"),
            )
            result["alternate_csv_sales_disagreements"] = int(
                alternate_join["Total_Sales_raw"].ne(alternate_join["Total_Sales_alternate"]).sum()
            )
            result["alternate_csv_positive_for_raw_nonpositive"] = int(
                (alternate_join["Total_Sales_raw"].le(0) & alternate_join["Total_Sales_alternate"].gt(0)).sum()
            )
    return result


def build_analytical_model(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    clean = standardize_tables(tables)
    transactions = clean["transactions"]
    customers = clean["customers"]
    if customers["Customer_ID_Masked"].duplicated().any():
        raise ValueError("Customer_ID_Masked must be unique in Customers_Clean for a many-to-one join.")

    profile = customers[["Customer_ID_Masked", "Customer_Name", "Email_Masked", "State"]]
    orders = transactions.merge(
        profile,
        on="Customer_ID_Masked",
        how="left",
        validate="many_to_one",
        indicator="Customer_Match",
    )
    if len(orders) != len(transactions) or orders["Order_ID"].nunique(dropna=True) != transactions["Order_ID"].nunique(dropna=True):
        raise ValueError("Customer enrichment changed transaction row or order cardinality.")

    orders["State"] = orders["State"].fillna("Unassigned")
    orders["Customer_Match"] = orders["Customer_Match"].map(
        {"both": "Matched", "left_only": "No customer ID match"}
    )
    orders = add_order_features(orders)
    customer_features, thresholds = build_customer_features(orders)

    marketing = clean["marketing"]
    inventory = clean["inventory"]
    relationship_rows = [
        {
            "Tables": "Transactions -> Customers",
            "Key": "Customer_ID_Masked",
            "Decision": "Validated many-to-one left join",
            "Evidence": f"{transactions['Order_ID'].nunique()} orders retained; {transactions['Customer_ID_Masked'].dropna().nunique()} distinct transaction IDs match the unique customer key",
        },
        {
            "Tables": "Transactions -> Marketing",
            "Key": "Campaign_ID",
            "Decision": "Kept separate",
            "Evidence": "Campaign_ID is not available in the transaction table; campaign-to-order attribution is unsupported",
        },
        {
            "Tables": "Transactions -> Inventory",
            "Key": "SKU / Product",
            "Decision": "Kept separate",
            "Evidence": "SKU and product fields are not available in the transaction table; product-level sales linkage is unsupported",
        },
    ]

    field_quality_rows = []
    for table_name in ("transactions", "customers", "marketing", "inventory"):
        frame = clean[table_name]
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            if missing:
                field_quality_rows.append(
                    {"Table": table_name.title(), "Field": column, "Missing": missing, "Rows": len(frame)}
                )

    quality = {
        "field_missing": pd.DataFrame(field_quality_rows, columns=["Table", "Field", "Missing", "Rows"]),
        "relationships": pd.DataFrame(relationship_rows),
        "transaction_rows": len(transactions),
        "order_ids": int(transactions["Order_ID"].nunique(dropna=True)),
        "duplicate_order_ids": int(transactions["Order_ID"].duplicated().sum()),
        "transaction_customer_ids_missing": int(transactions["Customer_ID_Masked"].isna().sum()),
        "customer_profiles": len(customers),
        "customer_ids_in_transactions": int(transactions["Customer_ID_Masked"].nunique(dropna=True)),
        "matched_customer_orders": int(orders["Customer_Match"].eq("Matched").sum()),
        "sales_missing": int(transactions["Total_Sales"].isna().sum()),
        "order_dates_missing": int(transactions["Order_Date"].isna().sum()),
        "marketing_rows": len(marketing),
        "marketing_dates_missing": int(marketing["Campaign_Date"].isna().sum()),
        "marketing_reach_missing": int(marketing["Reach"].isna().sum()),
        "marketing_ctr_missing": int(marketing["CTR"].isna().sum()),
        "inventory_rows": len(inventory),
        "inventory_stock_missing": int(inventory["Stock"].isna().sum()),
        "dated_sales_orders": int(orders.dropna(subset=["Order_Date", "Total_Sales"])["Order_ID"].nunique()),
        "date_min": orders["Order_Date"].min(),
        "date_max": orders["Order_Date"].max(),
        "source_reconciliation": audit_source_reconciliation(transactions),
    }
    comparisons = compare_customer_and_state_means(orders)
    return {
        **clean,
        "orders": orders,
        "customer_features": customer_features,
        "thresholds": thresholds,
        "quality": quality,
        "statistical_comparisons": comparisons,
    }
