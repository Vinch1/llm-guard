# Scanners Using Small/Medium Models (by Model Name)

This list is based on the model names containing tokens like "small", "base", "distil", "tiny", or "sm".
It reflects default models in the current codebase (not parameter counts).

## Input Scanners

- BanCode -> vishnun/codenlbert-sm (small)
  - File: llm_guard/input_scanners/ban_code.py
- BanTopics -> MoritzLaurer/roberta-base-zeroshot-v2.0-c (base/medium)
  - File: llm_guard/input_scanners/ban_topics.py
- PromptInjection -> protectai/deberta-v3-base-prompt-injection-v2 (base/medium)
  - Optional: protectai/deberta-v3-small-prompt-injection-v2 (small)
  - File: llm_guard/input_scanners/prompt_injection.py
- Language -> papluca/xlm-roberta-base-language-detection (base/medium)
  - File: llm_guard/input_scanners/language.py
- EmotionDetection -> SamLowe/roberta-base-go_emotions (base/medium)
  - File: llm_guard/input_scanners/emotion_detection.py
- Anonymize (NER) -> Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 (base/medium)
  - File: llm_guard/input_scanners/anonymize_helpers/ner_mapping.py

## Output Scanners

- Bias -> valurank/distilroberta-bias (distil/small)
  - File: llm_guard/output_scanners/bias.py
- MaliciousURLs -> DunnBC22/codebert-base-Malicious_URLs (base/medium)
  - File: llm_guard/output_scanners/malicious_urls.py
- NoRefusal -> ProtectAI/distilroberta-base-rejection-v1 (distil/base)
  - File: llm_guard/output_scanners/no_refusal.py
- Relevance -> BAAI/bge-base-en-v1.5 (base/medium)
  - Optional: BAAI/bge-small-en-v1.5 (small)
  - File: llm_guard/output_scanners/relevance.py
- FactualConsistency -> MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (base/medium)
  - File: llm_guard/output_scanners/factual_consistency.py
- Sensitive (PII) -> Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 (base/medium)
  - File: llm_guard/output_scanners/sensitive.py
- LanguageSame -> papluca/xlm-roberta-base-language-detection (base/medium)
  - File: llm_guard/output_scanners/language_same.py

## Not Classified by Name

Some scanners use ML models without a size token in the model ID (e.g., Gibberish, Toxicity, BanCompetitors).
If you want a strict size classification by parameter count, tell me the size thresholds.
