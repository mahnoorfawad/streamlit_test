import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st

st.title("Paraphrase Generator")
user_input = st.text_area("Enter a sentence for paraphrasing:")

model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

paraphrases = []

if st.button("Paraphrase"):
    if user_input:
        # Tokenize the user input
        input_ids = tokenizer.encode(user_input, return_tensors="pt", max_length=1024, truncation=True)

        # Generate paraphrases
        output = model.generate(
            input_ids,
            max_length=50,
            min_length=20,
            num_return_sequences=5,
            length_penalty=2.0,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        # Decode and store the paraphrases
        paraphrases = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

# Display the paraphrases
st.subheader("Paraphrased Versions:")
for i, paraphrase in enumerate(paraphrases):
    st.write(f"Paraphrase {i + 1}: {paraphrase}")