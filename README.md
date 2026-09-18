# Multimodal Speaker Identification

Identifying which of five speakers produced an utterance, using **audio features**, **text embeddings**, and **both fused together** — then using LIME and SHAP to explain what each model actually keyed on.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JanaEmad1/multimodal-speaker-identification/blob/main/multimodal_speaker_id.ipynb)

The headline result is a **negative one**, and that is the interesting part: adding the text modality did not improve over audio alone, and the notebook shows why.

## Dataset

[CMU Arctic](https://huggingface.co/datasets/MikhailT/cmu-arctic) via Hugging Face, restricted to 5 speaker subsets — `aew`, `ahw`, `aup`, `awb`, `axb` — merged into **4,049 utterances**. Each record carries an audio waveform, its transcription, and a speaker ID used as the classification label.

## Method

| Stage | Approach |
|---|---|
| Audio features | 13 MFCC coefficients (`torchaudio`), mean-pooled over time → `[13]` |
| Text features | `distilbert-base-uncased` embeddings, mean-pooled over tokens → `[768]` |
| Early fusion | Concatenate text + audio → `[781]`, classify with an MLP `(128, 64)` |
| Late fusion | Separate LogisticRegression per modality, combine predictions |
| Self-supervised variant | DistilBERT further pretrained with masked-language-modelling on the transcripts, then fine-tuned for speaker classification |
| Explainability | LIME on the audio classifier, SHAP on the text classifier |

## Results

> **Read the sample sizes before the accuracies.** The fusion experiments run on a 50-utterance subset (`shuffle(seed=42).select(range(50))`), split 40 train / **10 test**. At n=10 a single misclassification moves accuracy by 10 points, and only 4 of the 5 speakers appear in that test split at all. These numbers are directional, not conclusive.

| Setup | Accuracy | Test set |
|---|---|---|
| Audio only (MFCC → LogisticRegression) | 90.0 % | 10 |
| Text only (DistilBERT → LogisticRegression) | 20.0 % | 10 |
| Early fusion (MLP on 781-d vector) | 90.0 % | 10 |
| Late fusion | 90.0 % | 10 |
| DistilBERT + MLM self-supervision, fine-tuned | 26.9 % | ~810 |

### What these numbers actually say

1. **Text carries essentially no speaker identity — by construction.** Every CMU Arctic speaker reads *the same prompt sentences*. Two speakers therefore produce near-identical transcriptions, so a text model has nothing to separate them with. Text-only lands at 20.0 %, which is exactly chance for 5 classes.

2. **The properly-sized experiment confirms it.** The DistilBERT classifier was evaluated on the full held-out split (~810 utterances, not 10) and still reached only 26.9 % — barely above the 20 % chance line, despite MLM pretraining. That is the trustworthy version of the text-only result.

3. **Fusion did not beat audio alone.** Early fusion, late fusion, and audio-only all sit at 90 %. Concatenating 768 near-useless text dimensions onto 13 informative audio ones added nothing. Fusion only helps when both modalities carry complementary signal, and here the second one does not.

The honest summary: **speaker identity lives in the audio, not the transcript**, and this dataset is a good demonstration of when *not* to reach for a multimodal model.

## Explainability

- **LIME** (audio classifier) — on the inspected example the model predicts `aup` with 96 % confidence, driven by a small number of MFCC-derived features crossing their decision thresholds.
- **SHAP** (text classifier) — a `LinearExplainer` over the DistilBERT embedding features, consistent with the finding that no individual text dimension separates speakers well.

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook multimodal_speaker_id.ipynb
```

Or open it directly in Colab with the badge above. The dataset downloads automatically from Hugging Face on first run. Committed cell outputs let you read every result without executing anything.

## Limitations / what I would change

- **Evaluate on far more than 10 samples.** The 50-utterance subset was chosen for runtime; 4,049 utterances were available. This is the single biggest weakness.
- **Stratify the split** so every speaker is represented in the test set — `ahw` currently is not.
- **Use a dataset with speaker-specific text** if the goal is genuinely multimodal, or switch the text branch to something acoustically grounded (e.g. Wav2Vec2 embeddings) rather than transcript semantics.
- **Richer audio features.** 13 mean-pooled MFCCs discard all temporal structure; deltas or a pretrained speech encoder would be a stronger baseline.

## Repository

```
multimodal_speaker_id.ipynb   full pipeline, outputs included
requirements.txt              dependencies
LICENSE                       MIT
```

Coursework for Cognitive Computing (multimodality), College of Artificial Intelligence, El Alamein.

## License

MIT — see [LICENSE](LICENSE).
