# !pip install fastapi uvicorn transformers accelerate bitsandbytes -q

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 1. Initialize FastAPI
app = FastAPI(
    title="Domain Name Generator API",
    description="Generate domain names using Microsoft Phi-2 Zero-Shot",
    version="1.0"
)

# 2. Load Model and Tokenizer
model_name = "microsoft/phi-2"
print("Loading model... This may take a while.")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

print("Model loaded successfully.")

# 3. Safety Guardrail Function
banned_keywords = ["adult", "porn", "sex", "nude", "explicit"]

def is_safe_input(text: str) -> bool:
    return not any(word in text.lower() for word in banned_keywords)

# 4. Request and Response Schema
class DomainRequest(BaseModel):
    business_description: str

class DomainResponse(BaseModel):
    business_description: str
    domain_suggestions: list

# 5. Endpoint: Generate Domains
@app.post("/generate", response_model=DomainResponse)
def generate_domains(request: DomainRequest):
    description = request.business_description.strip()

    # Safety check
    if not is_safe_input(description):
        raise HTTPException(status_code=400, detail="Inappropriate content detected.")

    if description == "":
        raise HTTPException(status_code=400, detail="Business description cannot be empty.")

    # Prompt template
    prompt = f"Generate 3 creative domain names for this business:\nBusiness: {description}\nDomains:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.95, do_sample=True)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract domains using regex
    domain_pattern = r"\b[a-zA-Z0-9-]+\.[a-z]{2,}\b"
    domains = re.findall(domain_pattern, generated_text)

    # Take only first 3 suggestions, or fallback
    if not domains:
        domains = ["example1.com", "example2.com", "example3.com"]

    return DomainResponse(business_description=description, domain_suggestions=domains[:3])