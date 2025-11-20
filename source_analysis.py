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


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
def configure_date_axis():
    """Format dates for 7-day reports."""
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45, ha="right")


def plot_articles_per_day(df, output="results/articles_per_day.png"):
    counts = df.groupby("day").size()

    plt.figure()
    plt.plot(counts.index, counts.values, marker="o", linewidth=2, color=PRIMARY_COLOR)
    plt.title("Daily Article Count")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    configure_date_axis()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_topics_per_day(df, output="results/topics_per_day.png"):
    exploded = df.explode("topics")
    pivot = exploded.groupby(["day", "topics"]).size().unstack(fill_value=0)

    plt.figure()
    pivot.plot(kind="bar", stacked=True, colormap="tab20")
    plt.title("Topic Distribution Per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")

