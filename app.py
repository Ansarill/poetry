import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

checkpoint = 'rugpt3large_lora_results/checkpoint-50'

# model loading (cached for performance)
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = PeftModel.from_pretrained(base_model, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer

model, tokenizer = load_model()

# move model to GPU if available
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model.to(device)

# UI
st.set_page_config(page_title="Text Generation Demo", layout="centered")
st.title("rugpt3 Text Generation Demo")
st.markdown("Use fine-tuned rugpt3 model with PEFT to generate text from a prompt.")

# prompt input
prompt = st.text_area("Enter your prompt:", key="prompt_input", height=150)

# generation method
method = st.radio("GenerationStrategy", ["Beam Search", "Top-p Sampling"], key="method_select")

# generation params UI
generation_params = {}

if method == "Beam Search":
    st.markdown("### Beam Search Parameters")
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max Length", 10, 200, 40)
        num_beams = st.slider("Number of Beams", 1, 20, 10)
        num_return_sequences = st.slider("Return Sequences", 1, 10, 3)
    with col2:
        no_repeat_ngram_size = st.slider("No Repeat N-gram", 0, 10, 2)
        num_beam_groups = st.slider("Beam Groups", 1, 10, 2)
        diversity_penalty = st.slider("Diversity Penalty", 0.0, 5.0, 1.0)

    generation_params.update({
        "max_length": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": diversity_penalty,
        "early_stopping": True
    })

else:
    st.markdown("### Top-p Sampling Parameters")
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max Length", 10, 200, 40)
        temperature = st.slider("Temperature", 0.1, 5.0, 0.7)
    with col2:
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
        num_return_sequences = st.slider("Return Sequences", 1, 10, 1)

    generation_params.update({
        "max_length": max_length,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences
    })

# generate button
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        with st.spinner("Generating text..."):
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_params
            )
            end_time = time.time()

            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # display results
            st.markdown("###Generated Text"+ ('s' if len(generated_texts) > 1 else '') + ":")
            for i, text in enumerate(generated_texts):
                st.text_area(f"Result {i + 1}", value=text, height=200, key=f"result_{i}")

            elapsed_time = round(end_time - start_time, 2)
            st.success(f"Generation time: {elapsed_time} seconds")