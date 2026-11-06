# VC-ReID

Minimal reproducibility package for **verification-controlled interactive video person re-identification**.

The repository contains the code required for feature construction, projection learning, official-protocol evaluation, signed query updates, verification-noise analysis, and the additional LS-VID/TF-CLIP/PRID2011 experiments. It also contains the parsed Evidence-MARS metadata and protocol indices used by the experiments.

## Release scope

Included:

- core training, retrieval, signed-feedback, and evaluation code;
- parsed Evidence-MARS metadata for 8,298 train, 1,980 query, and 9,330 gallery tracklets;
- MARS protocol indices and annotation-audit statistics;
- compact JSON records supporting the reported MARS, LS-VID, and PRID2011 results.

Excluded:

- raw MARS, LS-VID, and PRID2011 images or videos;
- model checkpoints and cached visual/text features;
- experiment logs and machine-specific paths;
- raw vision-language-model responses;
- annotation and question-generation prompts.

Users must obtain the underlying benchmarks from their original providers and comply with the providers' terms.

## Evidence-MARS metadata

The files under `data/evidence_mars/` are gzip-compressed JSONL. Each row is tied to one official MARS tracklet and records its official index, split, identity/camera identifiers, sampled-frame provenance, a parsed appearance description, 15 structured appearance fields, and localized action intervals. The duplicated raw model response has been removed.

```python
import gzip, json

with gzip.open("data/evidence_mars/query.jsonl.gz", "rt", encoding="utf-8") as f:
    first_query = json.loads(next(f))
```

`data/evidence_mars/annotation_audit.json` reports schema validity and aggregate action/unknown-field counts. Files under `data/protocol/` reproduce the official train/query/gallery indexing without server-specific feature paths.

## Code layout

- `vc_reid/`: MARS feature construction, projection learning, retrieval, reranking, metrics, and noise analysis.
- `experiments/`: verification-only MARS, LS-VID, TF-CLIP, and PRID2011 entry points.
- `results/`: compact machine-readable records corresponding to the manuscript tables.

The language-question generation prompts are not part of this release. Verification-only experiments and the signed Rocchio-style update can be run independently of those prompts.

## Environment

The experiments were run with Python 3.10, PyTorch, torchvision, NumPy, SciPy, Pillow, tqdm, scikit-learn, and Hugging Face Transformers. Install versions compatible with the CUDA runtime available on the target machine.

## Signed update

For unit-normalized query and gallery embeddings, verification selects the update sign:

```text
confirm: q_(t+1) = normalize(q_t + alpha_t * g_1^t)
reject:  q_(t+1) = normalize(q_t - alpha_t * g_1^t + 0.3 * alpha_t * p_t)
alpha_t = alpha_0 * rho^t
```

The learned encoders and gallery embeddings remain fixed during interaction.

## Citation and data responsibility

Please cite the corresponding manuscript and the original MARS, LS-VID, PRID2011, and TF-CLIP papers when using this release. The metadata does not grant rights to redistribute the original benchmark media.
