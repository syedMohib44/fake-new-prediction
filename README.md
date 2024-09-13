# Fake News Detection with Linear SVC

![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange)
![pandas](https://img.shields.io/badge/pandas-2.2.2-yellow)
![numpy](https://img.shields.io/badge/numpy-2.1.1-green)
![MIT License](https://img.shields.io/badge/License-MIT-brightgreen)

This project uses a Support Vector Machine (SVM) classifier to detect fake news articles. It utilizes the `scikit-learn` library to vectorize text data with TF-IDF and train a Linear SVC model on a dataset of news articles labeled as either "REAL" or "FAKE."

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To install and run the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/username/fake-news-detector.git
    cd fake-news-detector
    ```

2. Clone the dataset:

    ```bash
    git clone https://huggingface.co/datasets/Trinisha/fake_or_real_news
    find file `fake_or_real_news` and put it in root directory of the project
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once installed, you can use the script to detect fake news by passing a CSV file of news articles. The CSV file should have two columns: `text` (the article text) and `label` (either "REAL" or "FAKE").

Here's how you can run the prediction:

```bash
python fake_news_detection.py
```

## Project Structures

```
.
├── fake_news_detection.py   # Main script for detecting fake news
├── requirements.txt         # Dependencies for the project
└── fake_or_real_news.csv    # Example dataset (should be provided)
```

## License

This project is licensed under the MIT License. See the [LICENSE](MIT License) file for details.
