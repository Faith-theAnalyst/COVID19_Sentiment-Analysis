from transformers import AutoModelForSequenceClassification, AutoTokenizer

import gradio as gr

model_path = f'Faith-theAnalyst/twitter_roberta_sentiment_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", padding='max_length',max_length=128)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    return {sentiment_classes[i]: float(probs.squeeze()[i]) for i in range(len(sentiment_classes))}
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

iface.launch(share=True)