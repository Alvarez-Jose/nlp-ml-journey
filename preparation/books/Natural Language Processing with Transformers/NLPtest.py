from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I'm excited to start grad school!")
print(result)
 