# [AAAI-2025] *AWRaCLe*: All-Weather Image Restoration using Visual In-Context Learning 

[Sudarshan rajagopalan](https://sudraj2002.github.io/) | [Vishal M. Patel](https://scholar.google.com/citations?user=AkEXTbIAAAAJ&hl=en)

[![Project Pager](https://img.shields.io/badge/Project-Page-blue)](https://sudraj2002.github.io/awraclepage/) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.00263)

<img src="./assets/intro.png" alt="" style="border:0; height:500px; width:1500px;">
<div class="content has-text-justified">
<p>
Fig. 1: Illustration of AWRaCLe: Our visual in-context learning approach for all-weather image restoration. Given a context pair (first two rows), AWRaCLe extracts 
relevant degradation context from it to restore a query image. Our method also performs selective removal of haze and snow from an image containing their mixture as shown in (d) and (e).
</p>
</div>
                        
<hr />

> **Abstract:** *All-Weather Image Restoration (AWIR) under adverse weather conditions is a challenging task due to the presence of different types of degradations. Prior research in this domain relies on extensive training data but lacks the utilization of additional contextual information for restoration guidance. Consequently, the performance of existing methods is limited by the degradation cues that are learnt from individual training samples. Recent advancements in visual in-context learning have introduced generalist models that are capable of addressing multiple computer vision tasks simultaneously by using the information present in the provided context as a prior. In this paper, we propose All-Weather Image Restoration using Visual In-Context Learning (AWRaCLe), a novel approach for AWIR that innovatively utilizes degradation-specific visual context information to steer the image restoration process. To achieve this, AWRaCLe incorporates Degradation Context Extraction (DCE) and Context Fusion (CF) to seamlessly integrate degradation-specific features from the context into an image restoration network. The proposed DCE and CF blocks leverage CLIP features and incorporate attention mechanisms to adeptly learn and fuse contextual information. These blocks are specifically designed for visual in-context learning under all-weather conditions and are crucial for effective context utilization. Through extensive experiments, we demonstrate the effectiveness of AWRaCLe for all-weather restoration and show that our method advances the state-of-the-art in AWIR.* 
<hr />

**Method**

<img src="./assets/block.png" alt="" border=0 height=500 width=1500></img>
<p>
AWRaCLe integrates degradation-specific information from a context pair to facilitate the image restoration process. 
Initially, CLIP features are extracted from the context pair and fed into Degradation Context Extraction (DCE) blocks at various levels of the decoder within the image restoration network. 
The Context Fusion (CF) blocks then fuse the degradation information obtained from the DCE blocks with the decoder features of the query image requiring restoration. Finally, the restored image is generated.
</p>


