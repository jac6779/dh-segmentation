# Dunnhumby Grocery Household Segmentation

Customer segmentation project using the **dunnhumby Complete Journey** grocery dataset to identify distinct household shopping patterns using **Python, scikit-learn, and unsupervised machine learning**.

🔗 **Live API:** https://ekcatmkhgt.us-east-1.awsapprunner.com/docs#/   

---

## Overview

This project segments grocery households based on differences in **shopping frequency, recency, basket size, and basket value**.

The workflow starts with more than 2.5 million transaction-level records, aggregates them into household-level behavioral features, explores how shopping behavior varies across the customer base, prepares a reduced feature set for clustering, and compares several unsupervised learning approaches.

The final stage profiles the resulting clusters using the original shopping measures and household demographic information so that statistical clusters can be interpreted as useful customer segments. The API also includes an OpenAI-powered text input route that converts a natural-language household description into the behavioral features used by the existing clustering pipeline.

This repo reflects a full customer segmentation workflow:

- transaction-level preprocessing
- household-level behavioral feature creation
- exploratory analysis
- feature selection and transformation
- feature scaling
- comparison of multiple clustering algorithms
- cluster evaluation
- customer and demographic profiling

---

## Business Case

Customer segmentation can help retailers better understand how different groups of customers shop.

Instead of treating every household the same, behavioral segments can support decisions around:

- customer targeting
- lifecycle and retention strategies
- promotional planning
- audience development
- personalized messaging
- differentiated customer experiences

The project was designed to answer a practical question:

> Can households be grouped into distinct customer segments based on how frequently they shop, how recently they visited, and the size and value of their typical shopping trips?

Demographic information is intentionally excluded from the clustering features and used afterward for profiling. This keeps the segments driven by actual shopping behavior while allowing demographic differences to help explain who the resulting customers are.

---

## Data Source

This project uses the **dunnhumby Complete Journey** dataset, a grocery retail dataset containing household purchase behavior across transactions, products, stores, campaigns, coupons, and customer demographics.

The primary modeling dataset was built from the transaction history.

Initial transaction data:

- **2,595,732 transaction-level rows**
- **2,500 households**
- **276,484 shopping baskets**
- approximately **711 days of shopping history**

Household demographic data is introduced after clustering for segment profiling rather than model training.

---

## End-to-End Workflow

### 1) Preprocessing

The preprocessing notebook converts the transaction-level data into one behavioral record per household.

Key household features created include:

- `num_trips`
- `recency_days`
- `total_spend`
- `avg_basket_value`
- `total_units_purchased`
- `avg_days_between_trips`
- `avg_basket_size`

Basket-level sales were first aggregated from the individual transaction records and then rolled up to the household level.

For example:

- **total spend** measures the total value purchased by each household
- **average basket value** measures the typical dollar value of a shopping trip
- **average basket size** measures the average number of units purchased per trip
- **recency** measures how recently the household last shopped
- **average days between trips** captures shopping frequency from another perspective

The resulting household dataset contains **2,500 customer records**.

Output artifact:

```text
data/clean_data/01_household_shopping.parquet
```

---

### 2) Exploratory Data Analysis

The goal of this analysis was to understand how household shopping behavior varies and whether the selected features are suitable for customer segmentation.

Many features are strongly right-skewed, with a small number of households showing much higher spending, frequency, or purchase volume than the rest.

Total spend generally increases with the number of trips, but households with similar trip counts can still spend very different amounts. Recency and average days between trips also appear to capture different aspects of engagement.

Log transformations improved the distributions of total spend and average basket value without removing high-value households. Total units purchased and average basket size contain especially large values, making their relationship difficult to interpret on the original scale.

Some features are also mathematically related, so their potential overlap was considered before clustering.

Overall, the data shows enough variation in spending, frequency, recency, and basket behavior to support household segmentation.

---

## Key EDA Findings

### Household shopping frequency

Shopping frequency varies substantially across the dataset.

- Median number of trips: **79**
- 75th percentile: **142.25**
- Maximum: **1,300**

This suggests a meaningful difference between occasional shoppers and households that visit very frequently.

### Recency

The median household last shopped only **6 days** before the end of the dataset, while some households had not returned for substantially longer periods.

This provides a different view of customer engagement than trip frequency alone.

### Total spend

Total spending is strongly right-skewed.

- Median total spend: **$2,157.75**
- 75th percentile: **$4,413.32**
- 95th percentile: **$9,754.14**
- 99th percentile: **$15,325.02**
- Maximum: **$38,319.79**

Households that shop more often usually spend more, although trip count does not explain all of the difference in total spending.

### Average basket value

Average basket value also varies considerably across households.

- Median basket value: **$27.42**
- 75th percentile: **$40.55**
- Maximum: **$165.83**

This helps distinguish households that may shop at a similar frequency but spend very different amounts during each trip.

---

### 3) Feature Engineering

The feature engineering stage reduces the household data to a smaller set of behavioral dimensions that capture distinct aspects of shopping behavior.

Final clustering features:

```text
num_trips
recency_days
avg_days_between_trips
log10_avg_basket_size
log10_avg_basket_value
```

These features represent:

| Feature | Behavioral Dimension |
|---|---|
| `num_trips` | Overall shopping frequency |
| `recency_days` | How recently the household shopped |
| `avg_days_between_trips` | Typical spacing between shopping trips |
| `log10_avg_basket_size` | Typical quantity purchased per trip |
| `log10_avg_basket_value` | Typical dollar value of a trip |

Average basket size and average basket value were log-transformed because of their strongly right-skewed distributions.

The clustering features were then standardized so that variables measured on different scales would contribute more evenly to the distance calculations used by the clustering algorithms.

`household_key` was retained separately as the household identifier and was not used as a clustering feature.

Output artifact:

```text
data/clean_data/03_model_features_processed.parquet
```

---

### 4) Clustering and Model Comparison

Four clustering approaches were compared:

- **KMeans**
- **Agglomerative Clustering**
- **Gaussian Mixture Model (GMM)**
- **DBSCAN**

Using multiple clustering approaches makes it possible to compare different assumptions about the structure of the customer population.

### KMeans

KMeans was used as the primary centroid-based baseline.

The elbow method was used to examine within-cluster sum of squares across different values of `k`, and the model was fit using:

```text
6 clusters
```

Evaluation results:

- **Silhouette Score:** 0.286
- **Davies-Bouldin Score:** 0.962
- **Calinski-Harabasz Score:** 791.571

### Agglomerative Clustering

Agglomerative clustering was used to compare the KMeans solution against a hierarchical approach.

The model used:

```text
6 clusters
Ward linkage
```

Evaluation results:

- **Silhouette Score:** 0.244
- **Davies-Bouldin Score:** 1.024
- **Calinski-Harabasz Score:** 592.657

### Gaussian Mixture Model

GMM was tested to allow households to be modeled using probabilistic cluster membership rather than strictly centroid-based assignment.

Evaluation results:

- **Silhouette Score:** 0.044
- **Davies-Bouldin Score:** 2.112
- **Calinski-Harabasz Score:** 261.430

### DBSCAN

DBSCAN was tested as a density-based alternative that can identify naturally dense customer groups while treating some households as noise.

Several combinations of `eps` and `min_samples` were evaluated.

The selected configuration used:

```text
eps = 1.25
min_samples = 5
```

This produced **3 clusters**, with approximately **2% of households classified as noise**.

Evaluation results on non-noise observations:

- **Silhouette Score:** 0.506
- **Davies-Bouldin Score:** 0.462
- **Calinski-Harabasz Score:** 61.835

---

## Model Results

| Model | Silhouette Score | Davies-Bouldin Score | Calinski-Harabasz Score |
|---|---:|---:|---:|
| DBSCAN | **0.506** | **0.462** | 61.835 |
| KMeans | 0.286 | 0.962 | **791.571** |
| Agglomerative Clustering | 0.244 | 1.024 | 592.657 |
| GMM | 0.044 | 2.112 | 261.430 |

Higher silhouette and Calinski-Harabasz scores are generally preferred, while a lower Davies-Bouldin score indicates better-separated clusters.

DBSCAN produced the strongest silhouette and Davies-Bouldin scores among the tested configurations, but its results are calculated on the non-noise households and it produces a different number of clusters than the other approaches.

KMeans produced a lower silhouette score but the strongest Calinski-Harabasz score and provides a straightforward six-segment structure.

Because customer segmentation ultimately needs to create groups that are both statistically defensible and useful to the business, model selection is not based on a single clustering metric alone.

---

## Cluster Profiling

The final stage of the project evaluates what the clusters actually represent.

Cluster labels are joined back to the original household shopping data so that segment characteristics can be interpreted using the original business-readable measures rather than standardized or log-transformed values.

The primary profiling variables are:

```text
num_trips
recency_days
avg_days_between_trips
avg_basket_size
avg_basket_value
```

For the two features that were log-transformed during modeling, the original versions are used during profiling so that results can be interpreted directly in units and dollars.

Household demographic information is then joined using `household_key`.

This makes it possible to evaluate questions such as:

- Which segments shop most frequently?
- Which groups have the largest or highest-value baskets?
- Which households appear more or less recently engaged?
- Do behavioral segments show meaningful differences in household composition, income, age, or other demographic characteristics?
- Are small clusters genuinely distinct customer groups or simply statistical edge cases?

The final clustering approach will be selected after considering both model quality and the usefulness of the resulting customer profiles.

---

## Why Demographics Were Excluded From Clustering

Demographic variables were intentionally reserved for the profiling stage.

The purpose of the model is to discover segments based on **observed shopping behavior**, not to predetermine customer groups using demographic characteristics.

This creates a useful separation:

```text
Shopping behavior → creates the segments
Demographics → helps explain the segments
```

That makes the resulting profiles easier to interpret for marketing and customer analytics use cases while keeping the clustering itself behavior-driven.

---

## Repository Structure

```text
dunnhumby-household-segmentation/
├── data/
│   ├── raw_data/
│   └── clean_data/
├── models/
├── notebooks/
│   ├── 01_dunnhumby_preprocessing.ipynb
│   ├── 02_dunnhumby_data_analysis.ipynb
│   ├── 03_dunnhumby_feature_engineering.ipynb
│   ├── 04_dunnhumby_modeling.ipynb
│   └── 05_dunnhumby_cluster_analysis.ipynb
├── README.md
└── requirements.txt
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Parquet
- Joblib

### Machine Learning Methods

- KMeans
- Agglomerative Clustering
- Gaussian Mixture Models
- DBSCAN
- StandardScaler
- log transformations
- silhouette analysis
- Davies-Bouldin index
- Calinski-Harabasz index

---

## Why This Project Matters

This project demonstrates:

- converting millions of transaction records into customer-level behavioral features
- exploratory analysis of grocery shopping behavior
- thoughtful feature selection for unsupervised learning
- handling highly skewed customer behavior without automatically removing valuable high-activity households
- comparing multiple clustering algorithms with different assumptions
- evaluating clustering with more than one statistical metric
- separating behavioral segmentation from demographic profiling
- translating unsupervised model output into customer segments that could support marketing and CRM decisions

Unlike a supervised prediction problem, there is no known target telling the model what the correct customer segments should be.

The main challenge is therefore not simply finding the clustering algorithm with the highest score. It is finding a segmentation that captures real differences in customer behavior and produces groups that can be understood and acted on.
