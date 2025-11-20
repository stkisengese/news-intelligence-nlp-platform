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


def plot_companies_per_day(df, output="results/companies_per_day.png"):
    df["org"] = df["org"].apply(normalize_organizations)
    exploded = df.explode("org")
    counts = exploded.groupby("day")["org"].nunique()

    plt.figure()
    plt.plot(counts.index, counts.values, marker="o", color=THIRD_COLOR, linewidth=2)
    plt.title("Unique Companies Mentioned Per Day")
    plt.xlabel("Date")
    plt.ylabel("Unique Company Count")
    configure_date_axis()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_sentiment_per_day(df, output="results/sentiment_per_day.png"):
    df["sentiment_label"] = df["sentiment"].apply(classify_sentiment)

    pivot = df.groupby(["day", "sentiment_label"]).size().unstack(fill_value=0)
    pivot = pivot[["Positive", "Negative"]]  # focus only on strong sentiment

    plt.figure()
    pivot.plot(kind="bar", color=[POS_COLOR, NEG_COLOR])
    plt.title("Positive vs Negative Sentiment Per Day")
    plt.xlabel("Date")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_top_companies(df, output="results/top_companies.png", top_n=20):
    df["org"] = df["org"].apply(normalize_organizations)
    exploded = df.explode("org")
    exploded = exploded[exploded["org"].str.strip() != ""]

    counts = exploded["org"].value_counts().head(top_n)

    plt.figure()
    counts.sort_values().plot(kind="barh", color=PRIMARY_COLOR)
    plt.title(f"Top {top_n} Most Mentioned Companies")
    plt.xlabel("Mentions")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_sentiment_per_company(df, output="results/sentiment_per_company.png", top_n=10):
    df["org"] = df["org"].apply(normalize_organizations)
    exploded = df.explode("org")

    top_companies = exploded["org"].value_counts().head(top_n).index
    subset = exploded[exploded["org"].isin(top_companies)]

    sentiment_avg = subset.groupby("org")["sentiment"].mean().sort_values()

    plt.figure()
    sentiment_avg.plot(kind="barh", color=SECONDARY_COLOR)
    plt.title(f"Average Sentiment for Top {top_n} Companies")
    plt.xlabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_topic_distribution(df, output="results/topic_distribution.png"):
    exploded = df.explode("topics")
    counts = exploded["topics"].value_counts()

    plt.figure()
    wedges, text, autotexts = plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        pctdistance=0.8,
        startangle=140,
        colors=plt.cm.Paired.colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )

    # Convert to donut
    plt.gca().add_artist(plt.Circle((0, 0), 0.5, color="white"))

    plt.title("Overall Topic Distribution")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


# ------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------
def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    df = load_enriched_data()
    if df is None:
        return

    plot_articles_per_day(df.copy())
    plot_topics_per_day(df.copy())
    plot_companies_per_day(df.copy())
    plot_sentiment_per_day(df.copy())
    plot_top_companies(df.copy())
    plot_sentiment_per_company(df.copy())
    plot_topic_distribution(df.copy())


if __name__ == "__main__":
    main()
