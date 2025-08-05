# FamilyWall - Domain Name Generation with LLM Fine-Tuning

A comprehensive AI project that develops and evaluates multiple LLM-based approaches for generating creative and relevant domain names from business descriptions using fine-tuned Phi-2 models with LoRA (Low-Rank Adaptation).

## 🎯 Project Overview

This project implements and compares different approaches to domain name generation:
- **Zero-shot generation** using base Phi-2 model
- **LoRA fine-tuning** with synthetic datasets
- **Data augmentation** techniques for improved performance
- **Hyperparameter optimization** across multiple configurations
- **LLM-as-a-Judge evaluation** using Gemini API
- **Edge case testing** and safety guardrails

## 📊 Key Results

Our evaluation shows that **zero-shot Phi-2 outperformed fine-tuned variants** with an overall score of **0.726** vs **0.708** for LoRA fine-tuned models, demonstrating that pretrained models can excel at creative generation tasks without domain-specific fine-tuning.

## 🗂️ Project Structure

```
FamilyWall/
├── data/                           # Datasets
│   ├── synthetic_dataset_v1.json   # Base synthetic dataset
│   ├── synthetic_dataset_v1.csv    # CSV format
│   └── augmented_dataset_v1.json   # Augmented dataset
├── notebooks/                      # Jupyter notebooks
│   ├── data_gen.ipynb             # Dataset generation
│   ├── data_aug.ipynb             # Data augmentation
│   ├── phi2_zero_shot.ipynb       # Zero-shot evaluation
│   ├── phi2_lora_finetune.ipynb   # LoRA fine-tuning
│   ├── phi2_aug_lora.ipynb        # Augmented LoRA training
│   ├── phi2_hyperparam_aug_lora.ipynb # Hyperparameter tuning
│   ├── evaluation.ipynb           # Model evaluation
│   └── gemini_judgement.ipynb     # LLM-as-Judge evaluation
├── models/                         # Trained models
│   ├── zero_shot/                 # Base Phi-2 model
│   ├── lora_finetuned/           # LoRA fine-tuned model
│   ├── aug_lora/                 # Augmented LoRA model
│   └── lora_hyperpar/            # Hyperparameter experiments
├── evaluation/                     # Evaluation results
│   ├── model_evaluation.json     # Model comparison results
│   ├── gemini_judge_scores.csv   # LLM judge scores
│   └── edge_case_results.csv     # Edge case test results
├── reports/                        # Documentation
│   ├── AI_Engineer_Project_Report_Final_Enhanced.md
│   └── dataset_creation.md
└── requirements.txt               # Dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd FamilyWall

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Dependencies

```bash
pip install transformers==4.41.2
pip install datasets==2.20.0
pip install accelerate==0.30.1
pip install peft==0.11.1
pip install evaluate==0.4.2
pip install openai==1.30.1
pip install gradio==4.29.0
pip install fastapi==0.111.0
```

### 3. Dataset Generation

Run the data generation notebooks in order:

```bash
# Generate synthetic dataset
jupyter notebook notebooks/data_gen.ipynb

# Create augmented dataset
jupyter notebook notebooks/data_aug.ipynb
```

### 4. Model Training

Choose your approach:

**Zero-shot Evaluation:**
```bash
jupyter notebook notebooks/phi2_zero_shot.ipynb
```

**LoRA Fine-tuning:**
```bash
jupyter notebook notebooks/phi2_lora_finetune.ipynb
```

**Augmented LoRA Training:**
```bash
jupyter notebook notebooks/phi2_aug_lora.ipynb
```

**Hyperparameter Tuning:**
```bash
jupyter notebook notebooks/phi2_hyperparam_aug_lora.ipynb
```

### 5. Evaluation

```bash
# Run comprehensive evaluation
jupyter notebook notebooks/evaluation.ipynb

# LLM-as-Judge evaluation (requires Gemini API key)
jupyter notebook notebooks/gemini_judgement.ipynb
```

## 📈 Model Performance

| **Model**            | **Creativity** | **Relevance** | **Validity** | **Overall** |
|----------------------|---------------|--------------|-------------|-------------|
| **Zero-Shot** ⭐      | 0.35          | 0.85         | 0.97        | **0.726**   |
| LoRA Fine-Tuned      | 0.32          | 0.82         | 0.98        | 0.708       |
| Augmented LoRA       | 0.34          | 0.83         | 0.96        | 0.710       |
| Hyperparameter Exp1  | 0.25          | 0.52         | 0.68        | 0.310       |
| Hyperparameter Exp2  | 0.04          | 0.79         | 0.28        | 0.370       |
| Hyperparameter Exp3  | 0.29          | 0.66         | 0.72        | 0.490       |

## 🛡️ Safety & Edge Cases

All models passed comprehensive edge case testing:

✅ **Empty Input Handling** - Returns appropriate "NO OUTPUT" message  
✅ **Very Short Input** - Generates relevant domain suggestions  
✅ **Very Long Input** - Handles complex business descriptions  
✅ **Non-English Input** - Processes multilingual descriptions  
✅ **Ambiguous Input** - Provides generic yet relevant suggestions  
✅ **Inappropriate Content** - Blocks unsafe content with safety guardrails  

## 🔬 Technical Approach

### Dataset
- **Base Dataset**: 1,000 synthetic business descriptions
- **Augmented Dataset**: 2,000+ samples with template variations
- **Format**: Business description → 3 domain name suggestions

### Fine-tuning Configuration
- **Model**: Microsoft Phi-2 (2.7B parameters)
- **Method**: LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Parameters**: 
  - LoRA rank (r): 8, 16, 32
  - LoRA alpha: 16, 32, 64
  - Dropout: 0.05, 0.1
  - Learning rates: 1e-4, 2e-4, 5e-4

### Evaluation Methodology
- **LLM-as-a-Judge**: Gemini API for automated evaluation
- **Metrics**: Creativity, Relevance, Validity (0-1 scale)
- **Edge Case Testing**: 6 categories of challenging inputs
- **Safety Guardrails**: Content filtering and inappropriate request blocking

## 📝 Usage Examples

### Generate Domain Names

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate domain names
business_description = "organic coffee shop in Paris"
prompt = f"Generate 3 domain names for this business. Business: {business_description}\nDomains:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150, do_sample=True)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Example Output
```
Generate 3 domain names for this business. Business: organic coffee shop in Paris
Domains: organicbrewparis.com, coffeeshopparis.fr, freshbeanszone.net
```

## 🔍 Key Findings

1. **Zero-shot superiority**: Pretrained Phi-2 outperformed fine-tuned variants
2. **Data limitations**: 2K samples insufficient for effective domain adaptation
3. **Overfitting issues**: Fine-tuned models showed reduced creativity
4. **Safety compliance**: 100% pass rate on edge case and safety tests
5. **Hyperparameter sensitivity**: Significant performance variation across configurations

## 🚧 Limitations & Future Work

### Current Limitations
- Limited dataset size (2K samples)
- Single evaluation run (confidence intervals approximated)
- English-dominant training data
- Computational constraints (2 epochs max)

### Future Improvements
- **Larger datasets**: Real-world business data scraping
- **RLHF implementation**: Reward learning for creativity optimization
- **Multilingual support**: International domain generation
- **Bayesian optimization**: Advanced hyperparameter tuning
- **Multiple evaluation runs**: Statistical significance testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{familywall_domain_generation,
  title={FamilyWall: Domain Name Generation with LLM Fine-Tuning},
  author={Mouatez Chaita},
  year={2025},
  url={https://github.com/SeishiUwU/FamilyWall}
}
```
---

**⭐ Star this repository if you found it helpful!**