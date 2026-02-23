# 📰 Financial News Sentiment Analysis

A Python tool that determines whether financial news headlines are positive, negative, or neutral using VADER sentiment analysis — with a clear visualisation of sentiment scores across multiple headlines.

## About Me

I am a Mechatronics Engineering graduate and a postgraduate in Finance from Henley Business School, looking to enter the field of finance in data science, machine learning, or quantitative finance.

- 💼 LinkedIn: [linkedin.com/in/sauravsen34](https://www.linkedin.com/in/sauravsen34)
- 📧 Email: saurav0sen34@gmail.com

---

## What Does This Project Do?

Sentiment in financial news moves markets. A single earnings headline or central bank announcement can shift asset prices significantly. This project automates the process of reading financial headlines and determining whether the tone is positive, negative, or neutral — a key input to many quantitative trading strategies.

---

## How It Works — VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool. It works by assigning pre-defined sentiment scores to words in its lexicon, then combining those scores to produce:

- **Positive score** — proportion of positive sentiment (0 to 1)
- **Negative score** — proportion of negative sentiment (0 to 1)
- **Neutral score** — proportion of neutral content (0 to 1)
- **Compound score** — overall sentiment from -1 (most negative) to +1 (most positive)

A compound score above 0.05 is classified as Positive, below -0.05 as Negative, and anything in between as Neutral — reflecting that small scores represent noise rather than meaningful signal.

---

## Key Limitation

VADER works at the word level and does not understand financial context. For example, "Federal Reserve raises interest rates" scores as neutral because the individual words are not inherently negative. A human finance professional immediately recognises this as bearish for equities — VADER does not.

The professional upgrade would be **FinBERT** — a version of the BERT transformer model trained specifically on financial news and earnings call transcripts. FinBERT understands domain-specific language and context, making it significantly more accurate for financial applications.

---

## How To Run

```bash
# 1. Clone the repository
git clone https://github.com/sauravsen3/sentiment-analysis.git
cd sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python sentiment.py
```

Output — printed table:
```
                                headline  compound sentiment
0   Apple reports record breaking qu...    0.6369  Positive
1   Federal Reserve raises interest ...    0.0000   Neutral
2   Tesla faces recall over safety c...   -0.4767  Negative
...
```

A horizontal bar chart is also saved as `sentiment_chart.png` — green for positive, red for negative, grey for neutral.

---

## Project Structure

```
sentiment-analysis/
│
├── sentiment.py        # Sentiment analysis and visualisation
├── requirements.txt    # Python dependencies
└── sentiment_chart.png # Generated on first run
```

---

## Tech Stack

- **NLTK / VADER** — rule-based sentiment scoring
- **pandas** — data structuring and output
- **matplotlib** — horizontal bar chart visualisation

---

## Next Step

Integrating a live news API (such as NewsAPI) to pull real headlines automatically, and upgrading the model from VADER to FinBERT for professional-grade financial sentiment classification.

---

*Part of a series of quantitative finance projects. Previous: Stock Return Direction Predictor. Next: Portfolio Optimisation.*
