# Q'anjob'al Machine Translation: Curriculum Learning and Transfer Learning for a Low-Resource Mayan Language

**Antara Chugh · UCLA Mars Lab**

[Read the background & motivation →](docs/BACKGROUND.md) · [Full references →](docs/REFERENCES.md) · [Project site →](#) *(coming soon)*

Indigenous migrant communities in the United States face significant barriers to language access, which limits civic engagement. Translation services for Indigenous migrant languages are hindered by limited human translators, government funding, and long turnaround times. This project evaluates post-training methods — supervised fine-tuning, continuous pre-training, and curriculum learning — to improve English-to-Q'anjob'al translation, one of many under-resourced Mayan languages spoken by large communities in Los Angeles and Champaign County, IL.

Related work: this project builds on findings from **["Impact Evaluation of AI-Language Access Models for Indigenous Language Communities in the City of Los Angeles"](https://ieeexplore.ieee.org/document/11098285)**, published at IEEE ETHICS 2025 (co-authored).

---

## Key Results

No open-source LLM has been explicitly trained on Q'anjob'al. Baselines on popular multilingual models (Gemma, Bloom, Aya, Qwen, Llama) performed near zero on translation out of the box. The best results came from **continuous pre-training (CPT) on a broader 13-language Mayan corpus, followed by Q'anjob'al-specific fine-tuning**:

| Model | Approach | BLEU | chrF |
|---|---|---|---|
| Gemma-3 4b | Base (zero-shot) | 0.0022 | 4.25 |
| Gemma-3 4b | Q'anjob'al-only SFT | 5.51 | 28.93 |
| **Gemma-3 4b** | **Mayan CPT → Q'anjob'al SFT** | **6.50** | **30.78** |
| Llama-3 1b | Base (zero-shot) | 0.0027 | 3.09 |
| Llama-3 1b | Q'anjob'al-only SFT | 3.65 | 25.74 |
| **Llama-3 1b** | **Mayan CPT → Q'anjob'al SFT** | **3.95** | **25.99** |

A key negative finding: stacking multiple multilingual sub-tasks (translation + structured language-family prompting) onto a small model caused training instability and **degraded** performance rather than helping — see [Model Training and Results](#model-training-and-results) below for the full breakdown across all experiments.

Full experiment writeups, tables, and takeaways for each stage (baseline, SFT, CPT, multi-stage fine-tuning, classification-based curriculum learning) are documented in the sections below.

## Datasets

Combines three public Q'anjob'al sources — the MayanV parallel corpus, Bloom Library children's book translations, and community-vetted University of Illinois educational materials — plus a 13-language Mayan parent corpus (~28k sentences) used for transfer-learning experiments. See [`data/README.md`](data/README.md) for provenance, licensing, and reproduction instructions. Raw data is not provided, only the scripts to fetch and rebuild it.

## Models

Published checkpoints:
- [Pre-trained Gemma-3 4b (Mayan CPT)](https://huggingface.co/atara15/gemma_mayan_qanjobal_continued_pretrain_gemma_4b)
- [Pre-trained Llama-3.2 1b (Mayan CPT)](https://huggingface.co/atara15/continued_pretrain_llama_3_2_1)

## Repo Structure

```
├── data/                          # dataset build scripts + data card
├── training_scripts/                       # CPT, SFT, LoRA scripts with eval 
│   ├── multi_stage_finetuning/   # knowledge-injection / curriculum experiments
|   ├── one_step_finetuning/      #SFT on English-Q'anjob'al Pairs
|   ├── pretraining_scripts/    
│   └── classification_based_curriculum_learning/ #Adding classification based SFT 
├── docs/
│   ├── BACKGROUND.md                  # full motivation, related work, literature review
│   └── REFERENCES.md                  # complete citation list

```

## Evaluation Metrics

- **BLEU** — precision of overlapping word n-grams between reference and candidate translation
- **chrF** — overlap of character sequences rather than whole words; more robust to synonyms, word roots, and gender than BLEU
- **Perplexity (PPL)** — how "surprised" the model is by held-out text; a score of 100 means the model is as uncertain as picking randomly among 100 equally likely words


## Hyperparameters

For all SFT/LoRA training runs:

```python
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

EPOCHS = 5
MAX_LEN = 512
BATCH = 4
GRAD_ACCUM = 4
LR = 2e-4
```

---

## Model Training and Results

### Baseline

No open-source LLM has explicitly been trained on Q'anjob'al, so baselines were established on popular open multilingual models. As expected, all performed poorly out of the box.

| Model | Dataset | BLEU, chrF |
|---|---|---|
| Gemma-2 9b | English-Q'anjob'al only | 0.0001 |
| Bloom-7b | English-Q'anjob'al only | 0.0001 |
| Cohere Aya-Expanse 8b | English-Q'anjob'al only | 0.0001 |
| Qwen3-4b | English-Q'anjob'al only | 0.0001 |
| **Gemma-3 4b** | **English-Q'anjob'al only** | **0.0022, 4.2506** |
| **Llama-3 1b** | **English-Q'anjob'al** | **0.0027, 3.0856** |

### Supervised Fine-Tuning

Models were fine-tuned using Structured Fine-Tuning (SFT) on the task "Translate this text from English to Q'anjob'al," using the Q'anjob'al text as the golden (target) response.

| Model | Dataset | BLEU Score, chrF Score |
|---|---|---|
| Gemma-2 9b | English-Q'anjob'al only | 0.1215 |
| Bloom-7b | English-Q'anjob'al only | 0.0216 |
| Cohere Aya-Expanse 8b | English-Q'anjob'al only | 0.023 |
| Qwen3-4b | English-Q'anjob'al only | 0.0099 |
| **Gemma-3 4b** | **English-Q'anjob'al only** | **5.5125, 28.9260** |
| **Llama-3 1b** | **English-Q'anjob'al** | **3.6511, 25.7399** |

### Continuous Pre-training on Mayan Corpora

Large-scale multilingual pretraining can substantially improve performance when parallel data is scarce (see [Background](docs/BACKGROUND.md) for the supporting literature). Here, the "higher-resource" pair is a combination of 13 other Mayan languages (Achi, Awakateko, Chuj, Itza', Ixil, Q'eqchi', Mam, Poqomam, K'iche', Sipakapense, Tektitek, Tz'utujil) — ~28k sentences, ~280k words — rather than a single higher-resource language, on the hypothesis that shared grammatical structure across the Mayan family would boost Q'anjob'al performance specifically.

Models were continuously pre-trained (CPT) on this broader Mayan corpus, then evaluated for perplexity on a held-out Q'anjob'al test set. Decreased perplexity indicates the model became more confident in predicting Q'anjob'al text, even before task-specific fine-tuning.


*CPT, Perplexity Results*

| Model | Base Model, Perplexity | CPT Model, Perplexity |
|---|---|---|
| Gemma-3 4b | 846.83 | 154.15 |
| Llama-3 1b | 330.94 | 84.39 |

The pretrained models were then trained on the SFT translation task, outperforming SFT applied to base models directly:

*CPT, SFT - Translation Task*

| Model | Datasets | BLEU | chrF |
|---|---|---|---|
| Pre-trained Gemma-3 4b | Mayan Corpora, English-Q'anjob'al | 6.4959 | 30.7793 |
| Pre-trained Llama-3 1b | Mayan Corpora, English-Q'anjob'al | 3.9450 | 25.9947 |

### Knowledge Injection in Prompts and Multi-Stage Fine-Tuning

This experiment tested whether stacking multiple multilingual sub-tasks onto Llama-3 1b, on top of already-limited Q'anjob'al data, would help or hurt translation quality. Three variants were compared: fine-tuning on generic Mayan translation before Q'anjob'al, adding a structured language-family/branch prompt to explicitly encode linguistic relationships, and running the same two-stage sequence on a model already continuously pre-trained on Mayan text.

Across all three, BLEU and chrF stayed low (0.02–0.05 — well below the 3.65 BLEU from Q'anjob'al-only SFT), and the structured-prompting variant's perplexity diverged to infinity, signaling training instability rather than improved language modeling. The takeaway: stacking multiple translation sub-tasks appears to cause overfitting and degrade actual Q'anjob'al performance versus simpler, single-task fine-tuning, perhaps because a small model like Llama-3 1b has limited capacity to absorb multiple multilingual objectives at once without sufficient data to support them.



*Too Many Multilingual Tasks → Overfitting, Degraded Performance*

| Model | Datasets | Intervention + Prompt | PPL | BLEU | chrF |
|---|---|---|---|---|---|
| Base Llama-3 1b | English-Mayan, English-Q'anjob'al | SFT: "Translate this text from English to Mayan:" → SFT: "Translate this text from English to Q'anjob'al:" | 103.20 | 0.024 | 8.55 |
| Base Mayan Llama-3 1b | English-Mayan, English-Q'anjob'al | SFT: "Language family: Mayan \| Branch: {branch} \| Language: {lang_name}\n Translate this text from English to {lang_name}:" → SFT: "Translate this text from English to Q'anjob'al:" | (inf) | 0.02 | 7.59 |
| Pre-trained Llama-3 1b | English-Mayan, English-Q'anjob'al | SFT: "Translate this text from English to Mayan:" → SFT: "Translate this text from English to Q'anjob'al:" | – | 0.05 | 8.62 |

### Curriculum Learning: Classification-Based Auxiliary Task

Building on the finding above, this experiment tested a different kind of curriculum step: an auxiliary **classification task** (identify which Mayan language a text belongs to) rather than another translation task. Llama-3 1b was continuously pre-trained on Mayan data, then supervised fine-tuned to classify Mayan languages, before finally being fine-tuned and evaluated on Q'anjob'al translation.

| Model | Classification Accuracy | BLEU | chrF |
|---|---|---|---|
| Llama 3 1b, Pretrained on Mayan Data | 0.00 | 0.0031 | 3.3824 |
| Llama 3 1b, Pretrained on Mayan Data, SFT to Classify Mayan Languages | 0.616 | 0.0735 | 7.0181 |

Adding the classification objective raised classification accuracy from 0% to 61.6%, and modestly improved translation BLEU (0.0031 → 0.0735) and chrF (3.38 → 7.02) over the CPT-only baseline — suggesting the model learned some transferable structure about how Mayan languages differ from one another. However, when this classification-tuned model was then fine-tuned specifically on Q'anjob'al translation, performance stalled well below direct Q'anjob'al SFT alone:

*Classification Model after SFT on Translation Task*

| BLEU | chrF |
|---|---|
| 1.0896 | 15.2068 |

**Takeaway:** unlike stacking multiple translation objectives (which actively degraded performance), the classification subtask mitigates, but does not eliminate, the instability seen elsewhere in the curriculum experiments. It helped the model build useful cross-lingual structure without collapsing training, but still fell short of focused, single-task Q'anjob'al fine-tuning. This shows added curriculum complexity doesn't reliably improve low-resource performance at this model scale, however, larger models may have greater capability to absorb multiple multilingual objectives at once and generalize across tasks. 

## Limitations

- BLEU/chrF are imperfect for Q'anjob'al given its non-standardized orthography and morphological complexity — scores should be read as directional, not absolute.
- Spanish–English pairs in MayanV and Bloom Library were machine-translated via the `Helsinki-NLP/opus-mt-es-en` model, introducing a layer of translation noise.
- Q'anjob'al training data remains small (~3k sentences) relative to typical MT benchmarks.

## Summary

Multilingual transfer learning is a promising avenue for improving Q'anjob'al translation — pre-training on broader Mayan corpora before specializing consistently helped larger models. But because Q'anjob'al data remains limited, stacking too many multilingual objectives at once destabilizes training and degrades performance, particularly on smaller models. The strongest results came from disciplined curricula (broad → narrow) rather than maximal multi-tasking.

---

For full background, motivation, and related-work literature review, see [`docs/BACKGROUND.md`](docs/BACKGROUND.md). For the complete citation list, see [`docs/REFERENCES.md`](docs/REFERENCES.md).