---
datasets:
- flytech/python-codes-25k
language:
- en
- tr
base_model:
- Qwen/Qwen3-0.6B
pipeline_tag: text-generation
library_name: transformers
tags:
- code
- text-generation-inference
---
## Hugging Face Model

The fine-tuned model is available on Hugging Face Hub:  
👉 [haydarkadioglu/Qwen3-0.6B-lora-python-expert-fine-tuned](https://huggingface.co/haydarkadioglu/Qwen3-0.6B-lora-python-expert-fine-tuned)

## Training Notebook (Google Colab)

You can reproduce the fine-tuning process or adapt it for your own dataset using the Colab notebook:  
👉 [Open in Google Colab](https://colab.research.google.com/drive/17mU5LFWT6JQ5uDI8FGGugyEkKnykw4Xj?usp=sharing)

# haydarkadioglu/Qwen3-0.6B-lora-python-expert

Qwen 0.6B LoRA fine-tuned for Python expert tasks

## Model Details
- **Model type:** Qwen 0.6B LoRA
- **Base model:** Qwen/Qwen-0.6B
- **Fine-tuned by:** @haydarkadioglu
- **Language(s):** English, Python


## Intended Use
- **Primary use case:** Code generation, Python expert help
- **Not suitable for:** General conversation, non-Python coding tasks

## Training Details
- **Dataset:** flytech/python-codes-25k
- **Steps / Epochs:** 3 epochs, batch size 8
- **Hardware:** A100 GPU / Colab T4
- **Fine-tuning method:** LoRA / PEFT

## Evaluation
| Step         | Training Loss |
| ------------ | ------------- |
| 100          | 1.8288        |
| 500          | 1.7133        |
| 1000         | 1.5976        |
| 1500         | 1.6438        |
| 2000         | 1.5797        |
| 2500         | 1.5619        |
| 3000         | 1.6235        |
| Final (3102) | **1.6443**    |

Final Results:
Training loss (avg): 1.64
Steps/sec: 0.645
Samples/sec: 10.3
FLOPs: 5.31e15

## Limitations
- The model might produce incorrect or insecure code.
- Not guaranteed to follow PEP8.
- May hallucinate libraries or functions.

## Example Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "haydarkadioglu/Qwen3-0.6B-lora-python-expert-fine-tuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Write a Python function, this function should return prime numbers between 0-100"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
