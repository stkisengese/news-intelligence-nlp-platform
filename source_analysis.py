import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import ast

# ------------------------------------------------------------
# GLOBAL STYLE CONFIGURATION (PROFESSIONAL STYLE)
# ------------------------------------------------------------
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["axes.titleweight"] = "bold"

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
THIRD_COLOR = "#2ca02c"
NEG_COLOR = "#d62728"
POS_COLOR = "#2ca02c"

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
def load_enriched_data(filepath="results/enhanced_news.csv"):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run enrichment script first.")
        return None

    df = pd.read_csv(filepath)

    # Convert list-string fields to real Python lists
    for col in ["org", "topics"]:
        df[col] = df[col].apply(ast.literal_eval)

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date

    return df


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------
def normalize_organizations(org_list):
    if not isinstance(org_list, list):
        return []

    normalized = []
    for org in org_list:
        if not isinstance(org, str):
            continue

        org = org.strip()

        # BBC normalization
        if org.startswith(("BBC News", "BBC Sport", "BBC Radio", "BBC")):
            normalized.append("BBC")
            continue

        # PSG/Paris Saint-Germain example
        if org.lower() in ["psg", "paris saint-germain"]:
            normalized.append("Paris Saint-Germain")
            continue

        normalized.append(org)

    return list(set(normalized))


# ------------------------------------------------------------
# SENTIMENT CLASSIFICATION
# ------------------------------------------------------------
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

