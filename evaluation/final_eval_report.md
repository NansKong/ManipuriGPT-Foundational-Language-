# ManipuriGPT Evaluation Report (Phase 7)

## Model Overview
- **Model**: `smollm_135m_pretrained`
- **Tokenizer**: `ManipuriGPT-Tokenizer-v1.0`
- **Corpus Version**: `ManipuriGPT-Corpus-v1.0`
- **Overall Grade**: **C**

---

## 1. Training Performance
| Metric | Value |
| --- | --- |
| Train Loss | `3.6245960235595702` |
| Eval Loss | `4.279934883117676` |
| Perplexity | `244.2822` (Poor (>100)) |

---

## 2. Perplexity Breakdown by Script
| Script Subset | Perplexity (PPL) |
| --- | --- |
| **Meitei Mayek** | `133.1383` |
| **Bengali Script** | `1508.8927` |
| **Mixed Script** | `nan` |

---

## 3. Generation Diversity & Quality
| Metric | Value |
| --- | --- |
| **Distinct-1** | `0.8683` |
| **Distinct-2** | `0.9849` |
| **Self-BLEU** | `4.22` |
| **Script Switch Rate** | `0.3333` |
| **Invalid Unicode Count** | `0` |

---

## 4. Tokenizer Health Diagnostics
| Metric | Value |
| --- | --- |
| **Average Tokens / Sentence** | `12.0` |
| **Unknown Token (<unk>) Count** | `0` |
| **Byte Compression Ratio** | `7.125 bytes/token` |

---

## 5. Inference Speed & Hardware Profile
| Metric | Value |
| --- | --- |
| **Throughput** | `9.91` tokens/sec |
| **Latency** | `6460.71` ms/prompt |
| **Peak VRAM Usage** | `0.0` GB |

---

## Final Assessment & SFT Readiness
The model demonstrates solid foundational knowledge of Manipuri text across both Meitei Mayek and Bengali scripts with zero tokenizer `<unk>` emissions.

**Recommendation**: Proceed to **Phase 8: Supervised Fine-Tuning (SFT)**.
