import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

# Sample financial headlines for demonstration
headlines = [
    "Apple reports record breaking quarterly revenue",
    "Federal Reserve raises interest rates again",
    "Tesla faces recall over safety concerns",
    "Microsoft acquires AI startup for $2 billion",
    "Markets tumble amid recession fears",
    "Goldman Sachs beats earnings expectations",
    "Oil prices surge on supply concerns",
    "Amazon announces mass layoffs",
]


def analyse_headlines(headlines):
    """
    Run VADER sentiment analysis on a list of financial headlines.
    Returns a DataFrame with scores and sentiment label for each headline.
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for headline in headlines:
        scores = analyzer.polarity_scores(headline)
        results.append({
            'headline': headline,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound'],
            'sentiment': 'Positive' if scores['compound'] >= 0.05
                        else 'Negative' if scores['compound'] <= -0.05
                        else 'Neutral'
        })

    return pd.DataFrame(results)


def plot_sentiment(df):
    """
    Plot compound sentiment scores as a horizontal bar chart.
    Green = Positive, Red = Negative, Grey = Neutral.
    """
    colours = ['green' if s == 'Positive'
               else 'red' if s == 'Negative'
               else 'grey' for s in df['sentiment']]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(df['headline'], df['compound'], color=colours)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Compound Sentiment Score")
    ax.set_title("Financial News Sentiment Analysis")
    plt.tight_layout()
    plt.savefig("sentiment_chart.png")
    plt.show()


if __name__ == "__main__":
    df = analyse_headlines(headlines)
    print(df[['headline', 'compound', 'sentiment']])
    plot_sentiment(df)
