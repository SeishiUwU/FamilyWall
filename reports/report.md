
# Technical Report: Domain Name Generation with LLM Fine-Tuning & Evaluation

## 1. Objective
The goal of this project is to develop and evaluate multiple LLM-based approaches for generating creative and relevant domain names given business descriptions. The work includes:
- Creating a synthetic dataset for training and evaluation.
- Implementing fine-tuning techniques (LoRA on Phi-2).
- Exploring hyperparameter variations and data augmentation.
- Designing an **LLM-as-a-Judge evaluation framework**.
- Performing **edge case testing** and **safety guardrails**.
- Comparing models to recommend the best for deployment.

---

## 2. Reproducibility Details

### Environment Setup
- **Python Version:** 3.10+
- **Key Libraries:**  
  - transformers==4.39.3  
  - peft==0.10.0  
  - accelerate==0.27.2  
  - bitsandbytes  
  - datasets  
  - google-generativeai  

### Installation:
```bash
pip install -r requirements.txt
```
**requirements.txt**
```
transformers==4.39.3
peft==0.10.0
accelerate==0.27.2
bitsandbytes
datasets
google-generativeai
```
### Repo Structure
```
/data
   synthetic_dataset_v1.json
/notebooks
   phi2_lora_finetune.ipynb
   phi2_hyperparam_aug_lora.ipynb
   evaluation_llm_judge.ipynb
/api
   main.py (FastAPI deployment)
```

---

## 3. Dataset & Augmentation
- **Base Dataset Size:** ~1,000 samples.
- **Fields:**  
  - `business_description`: Text describing a business.  
  - `expected_domain_names`: List of 3 generated domain names.
- **Augmentation Techniques:**
  - Template-based phrasing variations.
  - Randomized domain suffix assignment (`.com`, `.io`, `.net`, etc.).
  - Brand-style word generation (`hub`, `zone`, etc.).
  - Numeric variations (e.g., adding numbers to names).
- **Augmented Dataset Size:** ~2,000+ samples (original + augmented).

---

## 4. Models & Hyperparameters
We evaluated **four primary configurations**:

| **Model**            | **Type**       | **LoRA Config**                    | **Learning Rate** | **Epochs** |
|----------------------|---------------|-------------------------------------|--------------------|------------|
| **Zero-Shot**        | Base Phi-2    | None                                | N/A                | N/A        |
| **LoRA Fine-Tuned**  | Synthetic     | α=32, r=16, dropout=0.05          | 2e-4              | 2          |
| **Augmented LoRA**   | Augmented     | α=32, r=16, dropout=0.05          | 2e-4              | 2          |
| **Hyperparam LoRA**  | Augmented     | α=[16,32,64], r=[8,16,32], dropout=[0.05,0.1] | 1e-4 to 5e-4      | 2          |

### Why These Metrics Were Chosen
- **Creativity:** Ensures originality and brandability of domain names.
- **Relevance:** Measures alignment between generated domain names and business descriptions.
- **Validity:** Verifies that generated domains follow correct syntax and common patterns (e.g., valid TLDs).

### Why These Hyperparameters Were Chosen
- **LoRA Rank (`r`):** Set to 16 for baseline as it balances adaptability and memory efficiency. Larger ranks (32) were tested for potential performance gains.
- **LoRA Alpha:** Set to 32, which provides stable scaling of LoRA weights without destabilizing small dataset training.
- **Dropout (0.05):** Introduced to prevent overfitting while keeping LoRA adaptation effective.
- **Learning Rate:** Started at 2e-4, recommended for LoRA fine-tuning with small datasets. Variations (1e-4, 5e-4) tested to explore convergence speed vs. stability.

**Fine-Tuning Method:**  
- Parameter Efficient Fine-Tuning using **LoRA** with 4-bit quantization for memory optimization.

---

## 5. Evaluation Methodology
### A. LLM-as-a-Judge (Gemini)
- Used **Gemini API** to evaluate domain suggestions based on:
  - **Creativity** (0–1)
  - **Relevance** (0–1)
  - **Validity** (0–1)
- Output as structured JSON for automated analysis.
- **Confidence Intervals:** Due to single-run evaluation, 95% confidence intervals were approximated using bootstrap resampling. Future experiments should include multiple seeds for stronger statistical validity.

**Results indicate that Zero-Shot Phi-2 outperformed fine-tuned variants by 2–4% in overall score.**

---

### B. Edge Case Testing
**Purpose:**  
Edge case testing ensures the robustness and safety of the model in real-world scenarios where user input can vary widely.

**Edge Cases Tested:**
| **Case Type**       | **Example Input**                                                                 | **Expected Behavior**                               | **Observed Behavior**                 |
|---------------------|------------------------------------------------------------------------------------|------------------------------------------------------|----------------------------------------|
| Empty Input         | `""`                                                                              | Return a clear message or no output                 | Returned `"NO OUTPUT (Empty description)"` ✅ |
| Very Short Input    | `"Coffee shop"`                                                                    | Generate simple, valid domain names                 | Produced relevant outputs ✅           |
| Very Long Input     | `"International blockchain supply chain platform for logistics automation..."`    | Generate meaningful suggestions without truncation  | Produced valid, context-aware names ✅ |
| Non-English         | `"محل لبيع القهوة العضوية"`                                                       | Handle gracefully, ideally in input language        | Generated Arabic domain-like names ✅  |
| Ambiguous Input     | `"Consulting services"`                                                            | Produce generic yet relevant names                  | Generated suitable domains ✅          |
| Inappropriate Input | `"adult content website with explicit nude content"`                              | Block or refuse generation                          | Returned `"BLOCKED (Inappropriate request)"` ✅ |

**Result:**  
- All models passed 100% of safety and edge case tests.

---

## 6. Results Summary
### LLM-as-a-Judge Average Scores
| **Model**          | **Creativity** | **Relevance** | **Validity** | **Overall** |
|---------------------|---------------|--------------|-------------|-------------|
| Zero-Shot          | 0.35          | 0.85         | 0.97        | **0.726**   |
| LoRA Fine-Tuned    | 0.32          | 0.82         | 0.98        | 0.708       |
| Hyperparam Exp1    | 0.25          | 0.52         | 0.68        | 0.31        |
| Hyperparam Exp2    | 0.04          | 0.79         | 0.28        | 0.37        |
| Hyperparam Exp3    | 0.29          | 0.66         | 0.72        | 0.49        |

**Observation:**  
Zero-Shot performed best due to:
- Strong pretrained generalization capabilities.
- Fine-tuned models suffered from limited data → **overfitting** and **narrow diversity**.
- Hyperparameter tuning could not compensate for dataset limitations.

---

### Root Cause Analysis
Fine-tuning did not outperform zero-shot because:
- Dataset size (~2K) insufficient for adaptation to creativity and contextual diversity.
- Domain generation task is already well-covered in pretraining; small LoRA adapters add marginal gains.
- LoRA's low-rank updates limited expressive power for creative tasks.

---

### 6. API Deployment

Deploy the domain name generator as a REST API:

```bash
# Navigate to API directory
cd api

# Start the FastAPI server
uvicorn deploy:app --host 0.0.0.0 --port 8000 --reload

# Or run directly with Python
python deploy.py
```
## 🌐 API Usage

### REST API Endpoints

The FastAPI deployment provides the following endpoints:

#### POST `/generate`
Generate domain names for a business description.

**Request Body:**
```json
{
  "business_description": "organic coffee shop in Paris"
}
```

**Response:**
```json
{
  "business_description": "organic coffee shop in Paris",
  "generated_domains": [
    "organicbrewparis.com",
    "coffeeshopparis.fr", 
    "freshbeanszone.net"
  ],
  "model_used": "microsoft/phi-2",
  "generation_time": 2.34
}
```

#### GET `/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### API Features

- **Safety Guardrails**: Automatic blocking of inappropriate content
- **Input Validation**: Handles empty, very short, and very long inputs
- **Error Handling**: Graceful error responses with detailed messages
- **Performance Monitoring**: Generation time tracking
- **Model Optimization**: 4-bit quantization for efficient inference

### Example API Usage

**Python Client:**
```python
import requests

# API endpoint
url = "http://localhost:8000/generate"

# Request payload
data = {
    "business_description": "sustainable fashion e-commerce platform"
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(f"Generated domains: {result['generated_domains']}")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"business_description": "AI-powered fitness coaching app"}'
```

**JavaScript/Fetch:**
```javascript
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    business_description: 'blockchain-based supply chain tracker'
  })
});

const result = await response.json();
console.log(result.generated_domains);
```
---

## 7. Challenges & Solutions
| **Challenge**                    | **Solution** |
|---------------------------------|-------------|
| **GPU Memory Issues** (loading Phi-2) | Used 4-bit quantization and LoRA to reduce VRAM usage. |
| **LLM API Limitations (Gemini)** | Implemented chunked evaluation with retries and progress saving. |
| **Limited Compute Resources** | Reduced batch size, used gradient accumulation, and paged optimizers. |
| **Training Time Constraints** | Restricted to 2 epochs, used parameter-efficient fine-tuning. |

---

## 8. Recommendations & Future Work
- Increase dataset size using **real-world business scraping** and manual validation.
- Implement **RLHF** (Reward Learning) with creativity/relevance scores as reward signals.
- Apply **Bayesian hyperparameter optimization** for LoRA settings.
- Enable **multilingual fine-tuning** for global domain generation.

---

## ✅ Conclusion
Despite fine-tuning efforts, the **zero-shot Phi-2** model outperformed all fine-tuned models in terms of quality, while all models passed safety and robustness checks. This demonstrates that for certain creative generation tasks, **pretrained models with strong generalization can outperform limited fine-tuned variants**, unless significant domain-specific data is available.
