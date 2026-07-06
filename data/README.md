# Data Sources
 
 `data/` contains scripts to fetch and reconstruct the training data from original sources, plus documentation of provenance and licensing below.
 
 
---
 
## Data Sources
 
### 1. MayanV
 
- **Link:** https://github.com/transducens/mayanv
- **What it is:** The largest publicly available Mayan–Spanish parallel corpus, spanning 14 Mayan languages including Q'anjob'al (`kjb`) and Chuj (`cac`), built from official and community-produced materials.
- **Used for:**
  - The Q'anjob'al–Spanish portion, machine-translated (Spanish→English via local MarianMT, `Helsinki-NLP/opus-mt-es-en`) to build the English–Q'anjob'al training set.
  - The other 13 Mayan languages, used as the higher-resource "parent" corpus for transfer-learning experiments (~28k sentences, ~280k words total across all Mayan languages).
- **License:** See the [MayanV repository](https://github.com/transducens/mayanv) directly for current license terms before redistributing any derived data beyond research use.
- **Reconstruction:** `data/download_mayanv.py` (clone/fetch instructions) → `data/translate_mayanv_data.py` (translation) → `data/prepare_splits.py` (per-language train/test splits).
- **Data Processing:** Each language is labelled by the Mayan language branch it belongs to, as to run experiments on improving performance by training on only closely related Mayan languages. 
markdown![Graph showing Mayan Language Family Branches and History of Evolution](./Mayan_Language_Tree_in_colour%20(1).png)

### 2. Bloom Library — Q'anjob'al
 
- **Link:** https://bloomlibrary.org/language:kjb
- **What it is:** Publicly released children's book translations into Q'anjob'al, produced through work supported by the Guatemalan Ministry of Education and USAID. Provides naturalistic, non-domain-specific language, complementing MayanV's more formal register.
- **Used for:** Additional English–Q'anjob'al parallel sentences, merged into the final Q'anjob'al training set. Source texts were aligned across Q'anjob'al, Spanish, and English; Spanish–English pairs were translated the same way as MayanV.
- **License:** Bloom Library titles are released under a range of open licenses (commonly CC BY or similar) that vary **per book**. 
- **Reconstruction:** Individual titles must be downloaded.
### 3. University of Illinois Q'anjob'al Community Materials
 
- **What it is:** Publically available vocabulary lists, phrase sheets, and educational materials, can be found at https://clacs.illinois.edu/outreach/k-14-teaching-resources/languages
---
 
## Dataset sizes (as reported)
 
| Corpus | Sentences | Words |
|---|---|---|
| Q'anjob'al (MayanV + Bloom Library + UIUC materials, combined) | ~3,000 | ~30,000 |
| Broader Mayan parent corpus (12 non-Q'anjob'al MayanV languages) | ~28,000 | ~280,000 |
 
## Format
 
Processed training data is stored as JSONL, one example per line:
 
```json
{"mayan": "Waytzebʼil kʼal chi yun naq unin", "english": "The child is always sleepy", "lang_code": "kjb"}
```
Data split: 90% train; 10% test

## References 
- Sam Noble Museum. (2018). *Mayan Languages*. University of Oklahoma. https://samnoblemuseum.ou.edu/collections-and-research/ethnology/mayan-textiles/mayan-textiles-background/mayan-languages/
- [MayanV corpus](https://github.com/transducens/mayanv) — Curated Datasets and Neural Models for Machine Translation of Informal Registers between Mayan and Spanish Vernaculars. https://arxiv.org/pdf/2404.07673
- Center for Latin American & Caribbean Studies. (n.d.). *Languages*. University of Illinois Urbana-Champaign. https://clacs.illinois.edu/outreach/k-14-teaching-resources/languages
- Shosted, R. K. (UIUC). *Q'anjob'al in the Classroom: What K-12 Educators Might Learn from Linguists*. https://linguistics.illinois.edu/people/profile/rshosted
- Shosted, R. K., Maldonado, K., & Hallett, J. *Q'anjob'al Language Documentation*. https://linguistics.illinois.edu/people/profile/rshosted
- Juan, Andrés Mateo, & Hallett, J. (2009). *Tzib' yul ko Q'anjob'al* [Alphabet book]. Northeastern University Repository. https://repository.library.northeastern.edu/files/neu:334803
- Shosted, R. K. (UIUC). *Resources on the Q'anjob'al Language*. https://linguistics.illinois.edu/people/profile/rshosted
- Shosted, R. K. *How Can I Study Q'anjob'al?* https://linguistics.illinois.edu/people/profile/rshosted
- Mateo Pedro, P. *The Acquisition of Inflection in Q'anjob'al Maya*. Northeastern University Repository. https://repository.library.northeastern.edu/files/neu:1001741
- Bloom Library. (n.d.). *Q'anjob'al Language Collection*. SIL International. https://bloomlibrary.org/language:kjb
- Ministerio de Educación de Guatemala (MINEDUC). (n.d.). *Guatemala Ministry of Education Bookshelf*. Bloom Library, in partnership with USAID, Juárez & Associates, and SIL LEAD. https://bloomlibrary.org/Guatemala-MOE
- Tiedemann, J., & Thottingal, S. (2020). OPUS-MT — Building Open Translation Services for the World. *Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT 2020)*. Model used: `Helsinki-NLP/opus-mt-es-en`. https://huggingface.co/Helsinki-NLP/opus-mt-es-en






 
