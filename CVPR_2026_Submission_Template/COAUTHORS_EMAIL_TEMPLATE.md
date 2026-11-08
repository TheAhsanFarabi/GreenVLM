# Email Template for Co-Authors

**Subject:** Update & Next Steps: CVPR Submission Draft for GreenVLM ("The Illusion of 'Free' Vision in Real-World Multimodal Tasks")

**To:**  
Israt Khandaker, Nurul Labib, MR Saad, Ishayat Ibn Hasan, Fayez Abdullah, Sahriar Mahmud, Sakhor Satapdo  
**From:**  
Ahsan Farabi

---

Dear Team,

I hope this email finds you well.

I am pleased to share a major update on our research project challenging recent consensus on Vision-Language Model (VLM) energy consumption. We have established our official working paper title and completed the initial end-to-end draft in the official **CVPR 2026 submission template**.

### 📄 Paper Title & Scope
* **Title:** **GreenVLM: The Illusion of "Free" Vision in Real-World Multimodal Tasks**
* **Target Venue:** CVPR 2026 / ICCV 2027 / ACM MM 2027
* **Primary Subject Areas:** Vision & Language, Efficient & Scalable Vision, Evaluation & Benchmarking, Green AI

---

### 🔬 Core Empirical Breakthrough (Pilot Results)
Recent literature (Zhan et al., ACM MM 2026) claimed that *"seeing is free, speaking is not"*---arguing that autoregressive text generation accounts for 86%–97% of total inference energy, and that visual token pruning is futile ($\le$10% savings).

We completed a systematic 300-inference pilot benchmark across 4 distinct task families (DocVQA, ChartQA, POPE, and COCO) across 3 resolutions ($224^2, 448^2, 896^2$). Our results prove that their claim was an artifact of testing wordy image captions:
1. **The Inverted Bottleneck:** On real-world document OCR (DocVQA), visual data reasoning (ChartQA), and visual probing (POPE), **visual prefill consumes 98.9% to 100.0% of total inference energy**! Text decoding concludes in under 0.06 seconds ($<$1 Joule).
2. **First Joint Energy–Accuracy Pareto Frontier:** Scaling resolution from $224^2 \to 448^2$ buys a **+55% relative accuracy gain** in reading documents before hitting a distinct diminishing-returns elbow.

---

### 👥 Author List & Information Confirmation
Please check your name, affiliation, and email address below to confirm they are accurate for the CVPR submission template:

1. **Israt Khandaker** (`israt.khandaker@institution.edu`) — Affiliation: [Department/University]
2. **Nurul Labib** (`nurul.labib@institution.edu`) — Affiliation: [Department/University]
3. **Ahsan Farabi** (`ahsan.farabi@institution.edu`) — Affiliation: [Department/University]
4. **MR Saad** (`mr.saad@institution.edu`) — Affiliation: [Department/University]
5. **Ishayat Ibn Hasan** (`ishayat.hasan@institution.edu`) — Affiliation: [Department/University]
6. **Fayez Abdullah** (`fayez.abdullah@institution.edu`) — Affiliation: [Department/University]
7. **Sahriar Mahmud** (`sahriar.mahmud@institution.edu`) — Affiliation: [Department/University]
8. **Sakhor Satapdo** (`sakhor.satapdo@institution.edu`) — Affiliation: [Department/University]

*(If your preferred email or institution format differs, please reply with your updated details).*

---

### 🚀 Immediate Next Steps & Division of Work
1. **LaTeX Template & Codebase:**  
   The entire CVPR template (including introduction, 2025–2026 literature review, methodology, and pilot results) is live on our GitHub repository:  
   👉 **GitHub:** https://github.com/TheAhsanFarabi/GreenVLM  
   *(The template files are in `CVPR_2026_Submission_Template/` and can be imported directly into Overleaf).*
2. **Scaled GPU Experiments (Action Item for GPU Team):**  
   We have included turnkey GPU runners (`main_benchmark.py` and `COAUTHOR_GPU_HANDOFF.md`). We will scale the evaluations to dynamic-resolution models (`Qwen2.5-VL-7B` and `InternVL2.5-8B`) on the GPU cluster to observe quadratic visual token scaling ($196 \to 3136$ tokens).
3. **Draft Review:**  
   Please review the sections (especially the literature review in `sec/2_related_work.tex` and experimental results in `sec/4_experiments.tex`) and let me know your feedback.

Looking forward to our discussion and to finalizing this strong submission together.

Best regards,  
**Ahsan Farabi**  
Project Lead, GreenVLM
