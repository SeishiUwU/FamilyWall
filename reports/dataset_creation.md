# Synthetic Dataset Generation for Domain Name Suggestions

## Overview
This notebook generates a synthetic dataset mapping business descriptions to domain name suggestions. The dataset will be used for fine-tuning an LLM for the AI Engineer Homework assignment.

---

## Features
- **Diverse Categories**: Food, Tech, Education, Healthcare, Finance, etc.
- **Randomized Locations**: New York, Paris, Tokyo, etc.
- **Domain Suggestions**:
  - Uses cleaned keywords and brand-style name generation
  - Multiple domain suffixes (.com, .org, .io, etc.)
- **Safety Guardrails**: Filters out inappropriate descriptions
- **Edge Cases**:
  - Very short descriptions
  - Very long or complex descriptions
  - Ambiguous descriptions
- **Output Formats**: JSON and CSV for compatibility

---

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install pandas
   ```
2. Run the notebook in order. It will:
   - Generate synthetic descriptions
   - Create 3 unique domain name suggestions per description
   - Save the dataset in `../data/synthetic_dataset_v1.json` and `../data/synthetic_dataset_v1.csv`

---

## Key Functions
### `clean_keywords(text)`
Removes common stopwords and limits keywords to the first three meaningful words.

### `brandify(word)`
Creates brand-like names by appending suffixes (e.g., `ify`, `hub`, `zone`).

### `generate_domain_suggestions(description)`
Generates 3 domain name suggestions based on cleaned keywords or brand-style logic.

### `is_safe(description)`
Checks if the business description contains inappropriate terms like `adult`, `porn`, `gambling`, `violence`.

---

## Dataset Details
- **Fields**:
  - `business_description`: String (business idea)
  - `expected_domain_names`: List of 3 domain suggestions
- **Size**: 1000 examples (configurable)
- **Example Entry**:
  ```json
  {
    "business_description": "organic coffee shop in Paris",
    "expected_domain_names": [
      "organicbrewify.com",
      "coffeeshopzone.io",
      "freshbeanshub.net"
    ]
  }
  ```

---

## Outputs
- `../data/synthetic_dataset_v1.json`
- `../data/synthetic_dataset_v1.csv`

---