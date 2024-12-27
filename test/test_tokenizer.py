import json
from transformers import AutoTokenizer
# 你的 JSON 字符串
json_data = '''{
  "prompt": "How do I implement a TF-IDF based keyword extraction in Python?",
  "prompt_id": "123456",
  "messages": [
    {
      "content": "How do I implement a TF-IDF based keyword extraction in Python?",
      "role": "user"
    },
    {
      "content": "You can implement TF-IDF keyword extraction in Python using the `sklearn` library. Here's a basic example:\\n\\n1. Import necessary libraries:\\n```python\\nfrom sklearn.feature_extraction.text import TfidfVectorizer\\n```\\n2. Define your documents (e.g., a list of job descriptions):\\n```python\\ndocuments = ['Job description 1', 'Job description 2', ...]\\n```\\n3. Initialize and fit the TF-IDF vectorizer:\\n```python\\nvectorizer = TfidfVectorizer()\\nX = vectorizer.fit_transform(documents)\\n```\\n4. Extract the top keywords for each document:\\n```python\\nfeature_names = vectorizer.get_feature_names_out()\\nfor i in range(X.shape[0]):\\n    sorted_items = X[i].toarray().flatten().argsort()[::-1]\\n    top_keywords = [feature_names[j] for j in sorted_items[:5]]  # Adjust number of keywords\\n    print(f'Document {i+1} top keywords: {top_keywords}')\\n```\\nThis will give you the top 5 keywords for each document based on TF-IDF scores.",
      "role": "assistant"
    }
  ]
}'''

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"
data = json.loads(json_data)
encodeds = tokenizer.apply_chat_template(data["messages"], return_tensors="pt",padding=True,return_dict=True, max_length = 512)
encodeds_2 = tokenizer(data["messages"][0]["content"], return_tensors="pt", padding=True)
encodeds_3 = tokenizer(data["messages"][1]["content"], return_tensors="pt", padding=True)
print("shape:" , encodeds["input_ids"].shape)
print("input_ids:", encodeds["input_ids"])
print("input_ids2:", encodeds_2["input_ids"])
print("input_ids3:", encodeds_3["input_ids"])