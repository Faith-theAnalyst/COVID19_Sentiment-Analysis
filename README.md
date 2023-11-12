# NLP-Vaccination-Sentiment-Analysis: Decoding Twitter's Pulse on COVID-19 Vaccination

![COVID19](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/img_covid.png)

Welcome to the hub of sentiment analysis on Twitter's chatter about COVID-19 vaccinations! 🌐 This repository is your ticket to the Zindi NLP Challenge - "To Vaccinate or Not to Vaccinate," where we unravel the sentiments (positive, neutral, negative) hidden in Twitter posts discussing vaccination topics. Through a meticulous analysis of public sentiment, the solution aspires to assist public health organizations and policymakers in formulating impactful strategies for vaccine communication and promotion.

## Overview

The COVID-19 pandemic has brought about unprecedented global disruptions, triggering widespread social media discussions. This project delves into sentiment analysis of COVID-19-related Twitter conversations, focusing on vaccine-related topics. By analyzing public sentiment, the goal is to provide insights for public health organizations and policymakers. The project employs roBERTa and DistilBERT, prominent natural language processing models, and Gradio for creating an intuitive user interface.

## Summary

| Jupyter Notebook | Published Article | Link To Working App on Hugging Face |
| ----------------- | ------------------ | ----------------------------------- |
| [Notebooks](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/tree/main/Notebooks) | [Published Article](https://medium.com/@code.faith.17/to-vax-or-not-to-vax-decoding-twitters-pulse-on-the-covid-19-vaccine-acc6def10699) | [App Link](https://huggingface.co/spaces/Faith-theAnalyst/Covid19_Vaccine_Sentiment_App) |

## App Interface

Add the text you want to analyze and click on the **SUBMIT** button.

### Before Prediction

![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/app_interface.jpg)

### Negative Prediction
![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/neg_sentiment.jpg)

### Neutral Prediction
![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/neu_sentiment.jpg)

### Positive Prediction
![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/pos_sentiment.jpg)

## The Challenge: To Vax or Not to Vax?

As the race for a COVID-19 vaccine intensifies, understanding public sentiment is paramount. This challenge involves classifying Twitter posts into positive, neutral, or negative sentiments regarding vaccinations.

## Dataset

The dataset consists of labeled Twitter posts, each tagged with a sentiment label (-1 for negative, 0 for neutral, 1 for positive). Navigate the `Data` folder for:

- `Train.csv`: Labeled tweets for model training.
- `Test.csv`: Tweets for model testing.

## Approach

1. Data Preprocessing: Tokenization, lowercasing, removing special characters, etc.
2. Model Selection: Choosing a pre-trained Hugging Face transformer model.
3. Fine-Tuning: Training the model on the training data.
4. Validation: Evaluating the model on the validation set.
5. Gradio App: Creating a user-friendly interface for the model using Gradio.
6. Model Deployment: Uploading the model and pipeline to the HuggingFace platform.

## Getting Started

### 1. Installing the required packages

```bash
pip install -r requirements.txt
```

### 2. Follow the notebooks in the `notebooks` folder for step-by-step guidance.

## Usage

- Run the Jupyter notebooks in the `notebooks` folder to preprocess data, train the model, and evaluate its performance.
- Navigate to the `app` folder and run the Gradio app:

```bash
cd app
python app.py
```

## Sentiment Analysis Function

The `predict_tweet` function processes tweet text, obtains model predictions, and presents sentiment labels with confidence scores.

```python
# Sentiment Analysis Function
def predict_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", padding='max_length', max_length=128)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    return {sentiment_classes[i]: float(probs.squeeze()[i]) for i in range(len(sentiment_classes))}
```

## Gradio Interface Configuration

The Gradio interface defines how users interact with the model, offering an intuitive experience for the COVID-19 Vaccine Sentiment Classifier.

```python
# Gradio Interface Configuration
iface = gr.Interface(
    fn=predict_tweet,
    inputs="text",
    outputs="label",
    title="COVID-19 Vaccine Sentiment Classifier",
    description="Enter a text about vaccines to determine if the sentiment is negative, neutral, or positive.",
    examples=[
        ["Vaccinations have been a game-changer in public health, significantly reducing the incidence of many dangerous diseases and saving countless lives."],  
        ["Vaccinations are a medical intervention that introduces a vaccine to stimulate an individual’s immune response against a particular disease."],  
        ["Vaccines are rushed to the market without proper testing and are pushed by corporations that value profits over the well-being of the public."]
    ]
)
```

## Launching Gradio App

By invoking `iface.launch(share=True)`, the Gradio interface for the COVID-19 Vaccine Sentiment Classifier is initiated, facilitating easy sharing.

## Conclusion

This project explores sentiments in COVID-19 vaccine discussions, from fine-tuning models to creating a user-friendly Gradio app. The sentiment classifier acts as a conductor, unraveling insights into public perception. As the app comes to life, users are invited to explore sentiments within their own words, delving into the collective consciousness of vaccination discussions.

## Disclaimer

This readme provides a concise overview. For detailed exploration of code and implementation, refer to the code snippets in the text and visit the [GitHub repository](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis).

Feel free to explore, engage, and contribute to uncovering sentiments in the digital landscape of COVID-19 vaccine discussions. Contributions are welcome! Open issues and pull requests are encouraged.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. You can customize the repository structure, README content, and instructions based on your actual project progress and needs. Make sure to replace yourusername with your actual GitHub username in the repository clone link.














# NLP-Vaccination-Sentiment-Analysis: Decoding Twitter's Pulse on COVID-19 Vaccination
Welcome to the hub of sentiment analysis on Twitter's chatter about COVID-19 vaccinations! 🌐 This repository is your ticket to the Zindi NLP Challenge - "To Vaccinate or Not to Vaccinate," where we unravel the sentiments (positive, neutral, negative) hidden in Twitter posts discussing vaccination topics.Through a meticulous analysis of public sentiment, the solution aspires to assist public health organizations and policymakers in formulating impactful strategies for vaccine communication and promotion.


## Summary
|     Jupyter Notebook                       | Published Article|    Link To Working App on Hugging Face
| -------------                  | -------------    |    -----------------
|[Notebooks](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/tree/main/Notebooks)|  [Published Article](https://medium.com/@code.faith.17/to-vax-or-not-to-vax-decoding-twitters-pulse-on-the-covid-19-vaccine-acc6def10699)               |[App Link](https://huggingface.co/spaces/Faith-theAnalyst/Covid19_Vaccine_Sentiment_App)

## App Interface
Add the text you want to analyze and click on the **SUBMIT** button.

### Before Prediction

![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/app_interface.jpg)

### Negative Prediction
![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/neg_sentiment.jpg)

### Neutral Prediction
![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/neu_sentiment.jpg)

### Positive Prediction
![App Screenshot](https://github.com/Faith-theAnalyst/COVID19_Sentiment-Analysis/blob/main/Screenshots/pos_sentiment.jpg)

## The Challenge: To Vax or Not to Vax?
As the race for a COVID-19 vaccine intensifies, understanding public sentiment is paramount. This challenge involves classifying Twitter posts into positive, neutral, or negative sentiments regarding vaccinations.


## Dataset

The dataset consists of labeled Twitter posts, each tagged with a sentiment label (-1 for negative, 0 for neutral, 1 for positive). Navigate the `Data` folder for:
- `Train.csv`: Labeled tweets for model training.
- `Test.csv`: Tweets for model testing.

## Approach

1. Data Preprocessing: Tokenization, lowercasing, removing special characters, etc.
2. Model Selection: Choosing a pre-trained Hugging Face transformer model.
3. Fine-Tuning: Training the model on the training data.
4. Validation: Evaluating the model on the validation set.
5. Gradio App: Creating a user-friendly interface for the model using Gradio.
6. Model Deployment: Uploading the model and pipeline to the HuggingFace platform.

## Getting Started
### 1. Installing the required packages
pip install -r requirements.txt


### 2. Follow the notebooks in the `notebooks` folder for step-by-step guidance.


## Usage

- Run the Jupyter notebooks in the `notebooks` folder to preprocess data, train the model, and evaluate its performance.
- Navigate to the `app` folder and run the Gradio app:
cd app
python app.py


## Contributing

Contributions are welcome! Feel free to open issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
You can customize the repository structure, README content, and instructions based on your actual project progress and needs. Make sure to replace yourusername with your actual GitHub username in the repository clone link.


