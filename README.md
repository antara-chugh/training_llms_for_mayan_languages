
# Q'anjob'al Machine Translation: Curriculum Learning and Transfer Learning for a Low-Resource Mayan Language

## Introduction 
Indigenous migrant communities in the United States face significant barriers to language access, preventing civic engagement in these communities. Translation services for indigenous migrant languages are hindered due to limited human translators, government funding, and long waiting periods for translations. This work evaluates the potential of post-training methods to increase performance on English to Q’anjob’al translation, one of many Mayan languages. In this work, I build on previous work on transfer learning and curriculum learning techniques for low-resource languages. 


## Background and Summary of Impact Evaluation of AI-Language Access Models for Indigenous Language Communities in the City of Los Angeles Paper

### Q'anjob'al Language & Language Access Challenges

Q'anjob'al is a Mayan language originating in Guatemala, and is the parent language of many other Mayan languages, including Chuj, Akateko, and Popti', often referred to as Q'anjob'lan languages (Maya Q'anjob'al Interpreting Services, 2025). Due to historical persecution and migration, many Q'anjob'al speakers now reside in Mexico and the continental United States. In the US, the largest Q'anjob'al communities can be found in Los Angeles, California and Champaign County, Illinois, where communities struggle with civic engagement due to language access barriers (Zeng et al., 2025): though the City of Los Angeles issued a mandate to increase access to government materials in Q'anjob'al back in 2021, significant language barriers still exist, and the city is not able to translate government documents into Q'anjob'al at scale (Zeng et al., 2025). Nationally, Q'anjob'al is one of the most common languages spoken in immigration court, yet there remains a significant lack of human translators for these plaintiffs (Medina, 2019). These gaps have led to the spread of misinformation in Indigenous communities during the height of the COVID-19 pandemic, wrongful deportations, and the denial of asylum (Medina, 2019).

### Community-Driven Language Access Initiatives

Increasing language access for Indigenous migrant languages has primarily been driven by community and institutional efforts. In the city of Los Angeles, CIELO partnered with the USC Equity Research Institute to increase the amount of public data available for Indigenous migrant communities in the city. Through community surveys, the group identified 36 distinct Indigenous migrant languages spoken in Los Angeles (CIELO & USC Equity Research Institute, n.d.). Prompted by CIELO's efforts, the City of Los Angeles passed Executive Order 32, 2021 — Strengthening Language Access in the City of Los Angeles: Mandated Data, mandating increased data collection and translation for Q'anjob'al and other Indigenous migrant languages. The University of Illinois has established the Illinois Maya Institute to facilitate ethical engagement with Mayan communities and align research efforts. Q'anjob'al is further offered as coursework to students, and collaborations with community organizations, including Pixan' Konob', have helped create teaching resources to teach English speakers about Q'anjob'al (Center for Latin American & Caribbean Studies, n.d.-a). Finally, the Guatemalan Ministry of Education and USAID have worked to provide translations of children's books written in Spanish or English to Q'anjob'al, publicly available on the Bloom Library website (Bloom Library, n.d.; Ministerio de Educación de Guatemala, n.d.). 


## Datasets

The Q'anjob'al and Mayan dataset used for training combined three publicly available Q'anjob'al datasets — the MayanV parallel corpus (the largest Mayan-Spanish collection, adapted to English via Google Translate), Bloom Library children's book translations, and community-vetted educational materials from the University of Illinois. Refer to the `data` folder for instructions to reproduce the dataset and more information on the data used. 


## Model Training and Results

The following outlines the training strategies tested to improve performance on translation from English to Q'anjob'al. 

### Evaluation Metrics

The following metrics were used to test performance:
- BLEU : Measures the precision of overlapping chunks of words (n-grams) between a reference and candidate translation
- chRF: Overlap of character sequences (not whole words) between the model output and reference
- Perplexity (PPL): Measures how “surprised” the model is by the data; More precisely, PPL is a measure at each word, how many options the model choosing between -- a score of 100 indicates that the model is as uncertain as if picking randomly across 100 equally likely words 

Because BLEU scores measures precision at the word level, it can over-penalize the model when synonyms are used or when the model gets word roots or genders correct. chRF can account for this, with the assumption that synonyms often share roots and thus characters. 

### Baseline 

To the best knowledge, no open source large language model has explicitly been trained on Q'anjob'al. Thus, I started by establishing baselines on popular open source multilingual models. As expected, base models performed poorly on the translataion task. Baseline results are summarized in the following table.

| Model | Dataset | BLEU, chrF |
|---|---|---|
| Gemma-2 9b | English-Q'anjob'al only | 0.0001 |
| Bloom-7b | English-Q'anjob'al only | 0.0001 |
| Cohere Aya-Expanse 8b | English-Q'anjob'al only | 0.0001 |
| Qwen3-4b | English-Q'anjob'al only | 0.0001 |
| **Gemma-3 4b** | **English-Q'anjob'al only** | **0.0022, 4.2506** |
| **Llama-3 1b** | **English-Q'anjob'al** | **0.0027, 3.0856** |

### Hyperparamters 

Epochs: 5, Max_Len=512, Batch size 4, Learning Rate 2e-4
### Supervised Finetuning

To start, models were fine-tuned using Structured Fine-Tuning (SFT) on the task "Translate this text from English to Q'anjob'al," using the Q'anjob'al text as the golden (target) response.

*Finetuning Base Model*

| Model | Dataset | BLEU Score, chrF Score |
|---|---|---|
| Gemma-2 9b | English-Q'anjob'al only | 0.1215 |
| Bloom-7b | English-Q'anjob'al only | 0.0216 |
| Cohere Aya-Expanse 8b | English-Q'anjob'al only | 0.023 |
| Qwen3-4b | English-Q'anjob'al only | 0.0099 |
| **Gemma-3 4b** | **English-Q'anjob'al only** | **5.5125, 28.9260** |
| **Llama-3 1b** | **English-Qanjobal** | **3.6511, 25.7399** |


### Continuous Pre-training on Mayan Corpora

Work in low-resource NLP has shown that large-scale multilingual pretraining can substantially improve performance when parallel data is scarce. According to Zoph et al. (2016), training on a higher resource language pair can increase stability, since gradient updates on an unfamiliar, low-resource domain tend to be noisy on their own, and positive parameter transfer, since the model has already learned generalizable structure it can build on rather than starting from scratch. At sufficient scale, large multilingual pretraining also lets a model map similar concepts across different languages into similar regions of its embedding space — a property demonstrated by XLM-R, which significantly outperforms mBERT on major multilingual benchmarks and delivers particularly large gains for low-resource languages ([Conneau et al., 2020](https://aclanthology.org/2020.acl-main.747.pdf)).

In this case, the higher-resource language pair are the 13 other Mayan languages (Achi, Awakateko, Chuj, Itza, Ixil, Q'eqchi, Mam, Poqomam, K'iche, Sipakapense, Tektitek, Tz'utujil) of which there are 28k sentences, ~280,000 words. This differs than previous experiments, which typically use one, higher source language. Instead, I used 13 low resource languages combined as the higher resource pair, with the hypothesis that the shared grammar structure and roots between languages will boost overall performance on Q'anjob'al. 

Models were first continuously pre-trained (CPT) on the broader Mayan corpora, then evaluated for perplexity on a held-out Q'anjob'al test set. A decreased perplexity indicates the model has become more confident in predicting Q'anjob'al text, even before any task-specific fine-tuning.

Pretraining scripts can be found in the `pretraining_on_mayan_texts.py` script, 
the pre-trained models are available at Hugging Face. [Pre-trained Llama Model](https://huggingface.co/atara15/continued_pretrain_llama_3_2_1)
[Pre-trained Gemma Model](https://huggingface.co/atara15/gemma_mayan_qanjobal_continued_pretrain_gemma_4b)


*CPT, Perplexity Results*
| Model | Base Model, Perplexity | CPT Model, Perplexity |
|---|---|---|
| Gemma-3 4b | 846.83 | 154.15 |
| Llama-3 1b | 330.94 | 84.39 |



The pretrained models were trained on the SFT translation task, outperforming SFT on base models. 

*CPT, SFT - Translation Task*

| Model | Datasets | BLEU | chrF |
|---|---|---|---|
| Pre-trained Gemma-3 4b | Mayan Corpora, English-Q'anjob'al | 6.4959 | 30.7793 |
| Pre-trained Llama-3 1b | Mayan Corpora, English-Q'anjob'al | 3.9450 | 25.9947 |


### Knowledge Injection In Prompts and Multistage Finetuning

Next, I tested whether stacking multiple multilingual sub-tasks onto Llama-3 1b, on top of the already-limited Q'anjob'al data, would help or hurt translation quality. Three variants were compared: fine-tuning on generic Mayan translation before Q'anjob'al, adding a structured language-family/branch prompt to explicitly encode linguistic relationships, and running the same two-stage sequence on a model that had already been continuously pre-trained on Mayan text. Across all three, BLEU and chrF stayed low (0.02–0.05, well below the 3.65 BLEU achieved by Q'anjob'al-only SFT), and the structured-prompting variant's perplexity diverged to infinity — a sign of training instability rather than improved language modeling. The takeaway is that  the extra sub-tasks appear to have caused overfitting and degraded the model's actual Q'anjob'al performance compared to simpler, single-task fine-tuning, perhaps because a small model like Llama-3 1b has limited capacity to absorb multiple multilingual objectives at once without data to support them: rather than transferring generalizable structure.

Scripts can be found under the `multi_stage_finetuning` folder.

| Model | Datasets | Intervention + Prompt | PPL | BLEU | chRF |
|---|---|---|---|---|---|
| Base Llama-3 1b | English-Mayan, English-Q'anjob'al | SFT: "Translate this text from English to Mayan:" → SFT: "Translate this text from English to Q'anjob'al:" | 103.20 | 0.024 | 8.55 |
| Base Mayan Llama-3 1b | English-Mayan, English-Q'anjob'al | SFT: "Language family: Mayan \| Branch: {branch} \| Language: {lang_name}\n Translate this text from English to {lang_name}:" → SFT: "Translate this text from English to Q'anjob'al:" | (inf) | 0.02 | 7.59 |
| Pre-trained Llama-3 1b | English-Mayan, English-Q'anjob'al | SFT: "Translate this text from English to Mayan:" → SFT: "Translate this text from English to Q'anjob'al:" | – | 0.05 | 8.62 |

*Too Many Multilingual Tasks → Overfitting, Degraded Performance*



### Curriculum Learning

Building on the finding that stacking too many multilingual translation sub-tasks caused overfitting on a small model, this experiment tested a different kind of curriculum step: an auxiliary **classification task** rather than another translation task. Llama-3 1b was first continuously pre-trained on Mayan language data before finally being evaluated on Q'anjob'al translation. No SFT on the translation task was initially performed. 
| Model | Classification Accuracy | BLEU | chRF |
|---|---|---|---|
| Llama 3 1b, Pretrained on Mayan Data | 0.00 | 0.0031 | 3.3824 |
| Llama 3 1b, Pretrained on Mayan Data, SFT to Classify Mayan Languages | 0.616 | 0.0735 | 7.0181 |

Adding the classification objective raised classification accuracy from 0% to 61.6%, and modestly improved translation BLEU (0.0031 → 0.0735) and chrF (3.38 → 7.02) over the CPT-only baseline — suggesting the model learned some transferable structure about how Mayan languages differ from one another. However, when this classification-tuned model was then fine-tuned specifically on the Q'anjob'al translation task, performance stalled well below what direct Q'anjob'al SFT alone achieved:

| BLEU | chRF |
|---|---|
| 1.0896 | 15.2068 |

*Classification Model after SFT on Translation Task*

**Takeaway:** unlike stacking multiple translation objectives (which actively degraded performance), the classification subtask appears to mitigate — but not eliminate — the instability seen elsewhere in the curriculum experiments. It helped the model build some useful cross-lingual structure without collapsing training, but it still fell short of the BLEU/chrF achieved by focused, single-task Q'anjob'al fine-tuning, reinforcing that additional curriculum complexity doesn't reliably translate into better low-resource performance at this model scale.

Scripts can be found in the `classification_based_curriculum_learning` folder.


## References:
- Bloom Library. (n.d.). *Q'anjob'al Language Collection*. SIL International. https://bloomlibrary.org/language:kjb
- Center for Latin American & Caribbean Studies. (n.d.-a). *Q'anjob'al*. University of Illinois Urbana-Champaign. https://clacs.illinois.edu/graduate/qanjobal
- Center for Latin American & Caribbean Studies. (n.d.-b). *Languages*. University of Illinois Urbana-Champaign. https://clacs.illinois.edu/outreach/k-14-teaching-resources/languages
- CIELO (Comunidades Indígenas en Liderazgo) & USC Equity Research Institute. (n.d.). *CIELO Research*. https://mycielo.org/cielo-research/
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*. https://aclanthology.org/2020.acl-main.747.pdf
- Lou, R., et al. (2024). MayanV: Curated Datasets and Neural Models for Machine Translation of Informal Registers Between Mayan and Spanish Vernaculars. *arXiv preprint*. https://arxiv.org/pdf/2404.07673
- Maya Q'anjob'al Interpreting Services. (2025). *Discover Q'anjob'al*. https://www.mayaqis.com/discover-qanjobal
- Medina, J. (2019, March 19). Anyone Speak K'iche' or Mam? Immigration Courts Overwhelmed by Indigenous Languages. *The New York Times*. https://www.nytimes.com/2019/03/19/us/translators-border-wall-immigration.html
- Ministerio de Educación de Guatemala (MINEDUC). (n.d.). *Guatemala Ministry of Education Bookshelf*. Bloom Library, in partnership with USAID, Juárez & Associates, and SIL LEAD. https://bloomlibrary.org/Guatemala-MOE
- Sam Noble Museum. (2018). *Mayan Languages*. University of Oklahoma. https://samnoblemuseum.ou.edu/collections-and-research/ethnology/mayan-textiles/mayan-textiles-background/mayan-languages/
- transducens. (n.d.). *MayanV* [Source code repository]. GitHub. https://github.com/transducens/mayanv
- Zeng, W., et al. (2025). Impact Evaluation of AI-Language Access Models for Indigenous Language Communities in the City of Los Angeles. *2025 IEEE International Symposium on Ethics in Engineering, Science, and Technology (ETHICS)*. https://ieeexplore.ieee.org/document/11098285
- Zoph, B., Yuret, D., May, J., & Knight, K. (2016). Transfer Learning for Low-Resource Neural Machine Translation. *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016)*. https://aclanthology.org/D16-1163.pdf