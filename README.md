# ALIGNER: Learning Fine-grained Cross-modal Alignment for Text-Based Person Retrieval

## Overview

Text-Based Person Retrieval (TBPR) aims to retrieve pedestrian images based on natural language descriptions. A major challenge in this task is learning generalized and discriminative cross-modal representations for accurate alignment between visual and textual modalities. Existing methods often improve fine-grained representations through auxiliary supervision or global feature enhancement, but still face limitations in modeling precise local correspondences. To address this issue, we propose ALIGNER, a novel framework that focuses on learning many-to-many relationships between patch features to achieve fine-grained cross-modal alignment. Specifically, ALIGNER captures token-level semantic correspondences through discriminative correspondence learning, models high-level semantic consistency via cross-modal consistency learning, and improves representation robustness by incorporating granular feature modeling with local structural awareness. These components jointly enhance both generalization capability and fine-grained alignment performance. Extensive experiments on CUHK-PEDES, ICFG-PEDES, and RSTPReid demonstrate that ALIGNER achieves state-of-the-art performance in both standard and domain generalization settings.

---

## Framework

<p align="center">
  <img src="https://github.com/cceinhorn/ALIGNER/blob/main/figure/ALIGNER.png" width="900">
</p>

We provide model weights for reproducibility and further research. Click [here](https://pan.quark.cn/s/7d3184b2a7ae)

