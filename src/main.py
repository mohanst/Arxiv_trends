import time
import urllib.parse
import feedparser
import matplotlib.pyplot as plt
import pandas as pd


def fetch_arxiv_submissions(
    query: str, start_date: str, end_date: str, max_results: int = 1000
) -> pd.DataFrame:
    """Fetches papers from the arXiv API matching a Boolean query within a date window.

    Note: arXiv API requires uppercase Boolean operators (AND, OR, ANDNOT).
    """
    base_url = "http://export.arxiv.org/api/query?"
    start_dt = pd.to_datetime(start_date).tz_localize("UTC")
    end_dt = pd.to_datetime(end_date).tz_localize("UTC")

    records = []
    start = 0
    batch_size = 100
    encoded_query = urllib.parse.quote(query)

    print(f"Fetching papers for query: '{query}'...")

    while start < max_results:
        url = (
            f"{base_url}search_query={encoded_query}"
            f"&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        feed = feedparser.parse(url)

        if not feed.entries:
            break

        reached_earlier_bound = False
        for entry in feed.entries:
            pub_date = pd.to_datetime(entry.published)

            # Results are sorted descending; stop if we go past the start_date
            if pub_date < start_dt:
                reached_earlier_bound = True
                break

            if start_dt <= pub_date <= end_dt:
                records.append(
                    {
                        "id": entry.id,
                        "title": entry.title.replace("\n", " "),
                        "published": pub_date,
                        "category": entry.get(
                            "arxiv_primary_category", {}
                        ).get("term", ""),
                    }
                )

        if reached_earlier_bound or len(feed.entries) < batch_size:
            break

        start += batch_size
        time.sleep(3)  # Respect arXiv's polite 3-second API rate limit

    return pd.DataFrame(records)


def plot_submission_trend(
    df: pd.DataFrame, sub_interval: str = "ME", query_label: str = ""
):
    """Resamples paper submissions by a given sub-interval and plots a time series.

    sub_interval frequency options (Pandas aliases):
      'D'  : Daily
      'W'  : Weekly
      'ME' : Monthly
      'YE' : Yearly
    """
    if df.empty:
        print("No papers found matching the specified query and date range.")
        return

    df = df.copy()
    df.set_index("published", inplace=True)

    # Resample counts into the requested time bins
    counts = df.resample(sub_interval).size()

    interval_labels = {
        "D": "Daily",
        "W": "Weekly",
        "ME": "Monthly",
        "M": "Monthly",
        "YE": "Yearly",
        "Y": "Yearly",
    }
    freq_name = interval_labels.get(sub_interval.upper(), sub_interval)

    # Plot
    plt.figure(figsize=(11, 5.5))
    plt.plot(
        counts.index,
        counts.values,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        markersize=5,
    )

    title = f"{freq_name} arXiv Submissions Trend"
    if query_label:
        title += f"\nQuery: {query_label}"

    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("New Submissions", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: Papers in AI/ML mentioning transformers or LLMs, excluding computer vision
    search_query = "(cat:cs.GT) AND (ti:swap)"
    start_date = "2000-01-01"
    end_date = "2026-09-22"

    # 1. Fetch metadata
    df_papers = fetch_arxiv_submissions(
        query=search_query,
        start_date=start_date,
        end_date=end_date,
        max_results=1000,
    )
    print(f"Retrieved {len(df_papers)} matching submissions.")

    # 2. Plot time series aggregated by month ('ME')
    plot_submission_trend(
        df_papers, sub_interval="ME", query_label=search_query
    )