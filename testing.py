from transformers import AutoTokenizer, T5ForConditionalGeneration

model_path = "t5-small-hungarian-translator"
loaded_model = T5ForConditionalGeneration.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Then use it for translation
def translate(text):
    input_text = "translate English to Hungarian: " + text
    inputs = loaded_tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = loaded_model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test translation with a sample sentence
sample_text = "I love literature."
translated_text = translate(sample_text)    

print(f"English: {sample_text}")
print(f"Hungarian: {translated_text}")