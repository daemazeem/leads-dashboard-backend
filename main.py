from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model and data
CSV_PATH = "./dummy_leads_v2_with_icp.csv"  # Use the file with ICP reasons
MODEL_PATH = "./lead_conversion_model.joblib"

# Load Data
def get_df():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH)

@app.get("/api/leads")
async def get_leads():
    df = get_df()
    if df is None or not os.path.exists(MODEL_PATH):
        return {"error": "Data or model not found"}

    # Load trained pipeline
    pipeline = joblib.load(MODEL_PATH)

    # Active leads only
    active_leads_df = df[~df["status"].isin(["converted", "lost"])].copy()
    if active_leads_df.empty:
        return {"leads": []}

    # Prepare datetime features
    active_leads_df["lead_created_at_dt"] = pd.to_datetime(active_leads_df["lead_created_at"])
    active_leads_df["created_hour"] = active_leads_df["lead_created_at_dt"].dt.hour
    active_leads_df["created_dayofweek"] = active_leads_df["lead_created_at_dt"].dt.dayofweek

    # Feature columns
    feature_cols = [
        "platform", "utm_source", "utm_campaign", "utm_content",
        "campaign_id", "ad_id", "lead_created_timezone", "lead_created_city",
        "age", "gender", "language", "device_type", "operating_system", "browser",
        "created_hour", "created_dayofweek"
    ]

    # Predict conversion probability
    active_leads_df["conversion_probability"] = pipeline.predict_proba(active_leads_df[feature_cols])[:, 1]
    active_leads_df = active_leads_df.sort_values(by="conversion_probability", ascending=False)

    # Helper to format journey and durations
    def build_journey(row):
        journey = []
        durations = {}
        ts_new = pd.to_datetime(row.get("lead_created_at")) if pd.notna(row.get("lead_created_at")) else None
        ts_engaged = pd.to_datetime(row.get("replied_at")) if pd.notna(row.get("replied_at")) else None
        ts_booked = pd.to_datetime(row.get("call_booked_at")) if pd.notna(row.get("call_booked_at")) else None
        ts_converted = pd.to_datetime(row.get("converted_at")) if pd.notna(row.get("converted_at")) else None

        if ts_new:
            journey.append({"status": "new", "at": str(ts_new)})
        if ts_engaged:
            journey.append({"status": "engaged", "at": str(ts_engaged)})
            if ts_new:
                diff = (ts_engaged - ts_new).total_seconds() / 3600
                durations["new_to_engaged"] = f"{diff:.1f}h" if diff < 24 else f"{diff/24:.1f}d"
        if ts_booked:
            journey.append({"status": "booked", "at": str(ts_booked)})
            if ts_engaged:
                diff = (ts_booked - ts_engaged).total_seconds() / 3600
                durations["engaged_to_booked"] = f"{diff:.1f}h" if diff < 24 else f"{diff/24:.1f}d"
        if ts_converted:
            journey.append({"status": "converted", "at": str(ts_converted)})
            if ts_booked:
                diff = (ts_converted - ts_booked).total_seconds() / 3600
                durations["booked_to_converted"] = f"{diff:.1f}h" if diff < 24 else f"{diff/24:.1f}d"

        journey.sort(key=lambda x: str(x["at"]))
        return journey, durations

    # Use ICP reasons from CSV
    leads_data = []
    for _, row in active_leads_df.iterrows():
        journey, durations = build_journey(row)
        lead_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        lead_dict["journey"] = journey
        lead_dict["status_durations"] = durations
        # Use precomputed icp_reasons
        if "icp_reasons" in row and pd.notna(row["icp_reasons"]):
            try:
                import ast
                lead_dict["icp_reasons"] = ast.literal_eval(row["icp_reasons"])
            except:
                lead_dict["icp_reasons"] = []
        else:
            lead_dict["icp_reasons"] = []
        leads_data.append(lead_dict)

    return {"leads": leads_data}

@app.get("/api/analytics")
async def get_analytics():
    df = get_df()
    if df is None:
        return {"error": "Data not found"}

    # Convert dates
    date_cols = ["lead_created_at", "replied_at", "call_booked_at", "converted_at"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Funnel flags
    df["reached_reply"] = df["replied_at"].notna()
    df["reached_call"] = df["call_booked_at"].notna()
    df["reached_conversion"] = df["converted_at"].notna()

    total = len(df)
    replies = int(df["reached_reply"].sum())
    calls = int(df["reached_call"].sum())
    conversions = int(df["reached_conversion"].sum())

    funnel = [
        {"stage": "New Leads", "count": total, "rate": 1.0},
        {"stage": "Engaged (Reply)", "count": replies, "rate": round(replies/total, 3) if total else 0},
        {"stage": "Booked (Call)", "count": calls, "rate": round(calls/replies, 3) if replies else 0},
        {"stage": "Converted", "count": conversions, "rate": round(conversions/calls, 3) if calls else 0}
    ]

    # Segmented analysis
    def get_segmented_funnel(column, limit=None):
        segments = df.groupby(column).agg(
            total=("lead_id", "count"),
            replies=("reached_reply", "sum"),
            calls=("reached_call", "sum"),
            conversions=("reached_conversion", "sum")
        ).reset_index()
        segments = segments[segments["total"] > 3]
        segments = segments.sort_values(by="total", ascending=False)
        if limit:
            segments = segments.head(limit)
        result = []
        for _, row in segments.iterrows():
            result.append({
                "name": str(row[column]),
                "total": int(row["total"]),
                "reply_rate": round(row["replies"]/row["total"], 3),
                "call_rate": round(row["calls"]/row["replies"], 3) if row["replies"] > 0 else 0,
                "conv_rate": round(row["conversions"]/row["calls"], 3) if row["calls"] > 0 else 0
            })
        return result

    # Age binned
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 40, 60, 100], labels=["<25", "25-40", "41-60", "60+"])

    return {
        "funnel": funnel,
        "by_platform": get_segmented_funnel("platform"),
        "by_device": get_segmented_funnel("device_type"),
        "by_city": get_segmented_funnel("lead_created_city", limit=15),
        "by_age": get_segmented_funnel("age_group"),
        "by_campaign": get_segmented_funnel("utm_campaign", limit=10)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
