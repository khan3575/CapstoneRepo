# Presentation Speech — 5 Persons
## "Efficient Brain Tumour Segmentation using Graph Neural Networks on BraTS Datasets"
### BUBT Capstone — Supervisor: Mr. Shamim Ahmed

---

---

## PERSON 1 — Kishor Kumar Das (Slides 1–8)

---

**[Slide 1 — Title Slide]**

Good morning, everyone.
My name is Kishor Kumar Das, and I am here today with my team to present our capstone project.
Our project is titled: **"Efficient Brain Tumour Segmentation using Graph Neural Networks on BraTS Datasets."**
We have five team members — myself, Rifa Sanjida, Kishor Kumar Das, Md. Mahamudul Hasan, and Md. Minhajur Rahman.
Our supervisor is Mr. Shamim Ahmed, from the Department of Computer Science and Engineering at BUBT.
Let me start by giving you a broad overview of what we worked on and why it matters.

---

**[Slide 2 — Introduction / Background]**

Brain tumours are one of the most serious medical conditions a person can face.
Detecting them early and accurately can make the difference between life and death.
Doctors use MRI scans to study the brain and identify tumour regions.
But these scans produce enormous amounts of data, and analysing them manually takes a huge amount of time.
So there is a strong need for automated methods that can help doctors by quickly and accurately identifying where the tumour is.
This is what we tried to solve.

---

**[Slide 3 — Problem Statement]**

The existing deep learning methods, especially the popular CNN and U-Net based approaches, do solve this problem to some degree.
But they come with a big drawback — they are extremely heavy computationally.
They require very powerful, expensive GPUs with a lot of memory.
They are slow during inference.
This makes them very difficult to use in hospitals that do not have high-end hardware.
So our question was: can we build a system that is accurate AND lightweight at the same time?
That is the core problem we addressed in this project.

---

**[Slide 4 — Research Gap]**

When we looked at the existing literature, we found that almost all state-of-the-art methods focus only on accuracy.
Nobody was paying much attention to efficiency — to memory usage, inference time, and model size.
Methods like nnU-Net, SwinUNETR, and TransBTS achieve very high Dice scores, but they need 16 to 32 gigabytes of GPU memory.
They have hundreds of millions of parameters.
This is not realistic for most hospitals.
We saw this as a gap that needed to be filled.
Our goal was to match their accuracy as closely as possible, while being dramatically more efficient.

---

**[Slide 5 — Proposed Approach Overview]**

To solve this, we proposed a Graph Neural Network based pipeline.
Instead of feeding the full MRI volume directly into a heavy model, we first convert each MRI slice into a compact graph.
The key idea is to use something called superpixels — small, meaningful regions inside the brain MRI — as the nodes of the graph.
This lets us represent the entire brain in a very compact way.
Two connected slices become one small graph with around 92 nodes and 180 edges.
Compare that to millions of voxels in a raw MRI.
This compression is the foundation of why our method is so efficient.

---

**[Slide 6 — Research Objectives]**

We had four clear research objectives for this project.

First — to develop a complete GNN-based segmentation pipeline using superpixel graphs built from multi-modal BraTS MRI data.

Second — to achieve competitive Dice scores compared to CNN and U-Net baselines, proving that a lightweight model can still be accurate.

Third — to dramatically reduce computational requirements: specifically, memory usage, inference time, and parameter count.

And fourth — to demonstrate cross-dataset generalisation, meaning we wanted to test our model on a completely different dataset — BraTS 2023 — without any retraining, and see if it still performed well.

These four objectives guided every design decision we made.

---

**[Slide 7 — Dataset: BraTS]**

We used the BraTS datasets for this project.
BraTS stands for Brain Tumour Segmentation Challenge.
It is the standard benchmark dataset used by researchers all over the world for this task.
For training and evaluation, we used BraTS 2021, which contains MRI scans from 1,251 patients.
Each patient's scan has four MRI modalities — T1, T1CE, T2, and FLAIR.
Together they give us a comprehensive picture of the brain and the tumour.
We used 1,000 patients for cross-validation and kept 251 patients sealed as a held-out test set that was never touched during training.
For generalisation testing, we used the BraTS 2023 dataset — a completely separate dataset with different acquisition protocols.

---

**[Slide 8 — MRI Modalities]**

Let me briefly explain what these four MRI modalities mean.
T1 gives us the basic anatomical structure of the brain.
T1CE, which stands for T1 Contrast Enhanced, uses a contrast agent to highlight areas where the blood-brain barrier is broken — this is exactly where active tumour tissue tends to appear.
T2 highlights water in tissues — it makes oedema and swelling very visible.
And FLAIR suppresses normal cerebrospinal fluid, making abnormal tissue around the tumour clearer.
Each modality shows us something different about the brain.
By combining all four, we give the model a much richer picture of what is happening.
This concludes my part. I will now hand over to Rifa Sanjida, who will walk you through our methodology in detail.


---

---

## PERSON 2 — Rifa Sanjida (Slides 9–14)

---

**[Slide 9 — Methodology Overview]**

Thank you, Kishor.
My name is Rifa Sanjida, and I will explain how our system is built.
Our pipeline has several steps — from raw MRI data all the way to a final segmentation mask.
Let me walk you through each step carefully.

---

**[Slide 10 — SLIC Superpixels]**

The first step is to take each 2D axial slice of the MRI and divide it into small, compact regions called superpixels.
We use an algorithm called SLIC — Simple Linear Iterative Clustering.
We apply it on the T1CE channel, because that channel shows the tumour boundary most clearly.
We set the target number of superpixels to 200 per slice.
In practice, after the algorithm runs, we get around 46 superpixels per slice.
Each superpixel is a small, meaningful region of the brain.
This is the key step that gives us our huge spatial compression.
A single patient's full MRI volume has about 8.9 million voxels.
After superpixel conversion, we represent the same patient with only around 3,909 nodes.
That is a compression ratio of 2,284 times.

---

**[Slide 11 — Graph Construction]**

Once we have the superpixels, we build a graph.
We take two consecutive axial slices and combine them into one graph unit.
This gives us approximately 92 nodes and 180 edges per graph.
The edges connect superpixels that are related to each other.
We have two types of edges.
The first type is intra-slice edges — these connect superpixels within the same slice if they share a boundary. This is called a Region Adjacency Graph.
The second type is inter-slice edges — these connect superpixels across the two slices using a k-nearest-neighbours approach, where k equals 3. We only connect superpixels across slices if they overlap by more than 10% or their centroids are within 10 millimetres of each other.
This paired-slice structure allows the model to learn from spatial relationships both within a slice and across neighbouring slices.

---

**[Slide 12 — Node Features]**

Each node in the graph represents one superpixel.
We describe each superpixel using 15 features, organised into three groups.

The first group is intensity features — 8 features total.
For each of the four MRI modalities (T1, T1CE, T2, FLAIR), we compute the mean and standard deviation of pixel intensities inside the superpixel.

The second group is spatial features — 4 features.
These are the area of the superpixel, the normalised area relative to the slice size, and the y and x centroid coordinates.

The third group is morphological features — 3 features.
These are the perimeter of the superpixel, its compactness, and the intensity range.

Together these 15 features give the model enough information to understand both what the tissue looks like and where it is located.

---

**[Slide 13 — GraphSAGE Architecture]**

For the GNN model, we use GraphSAGE — Graph Sample and Aggregate.
We chose GraphSAGE because it is an inductive method.
This means it can generalise to graphs it has never seen before.
This is exactly what we need for cross-dataset testing on BraTS 2023.

Our GraphSAGE model has 5 layers.
Each layer has 256 hidden channels.
The output dimension is 64.
The total number of parameters is only 439,041.
That is less than half a million parameters.
Each layer uses mean-pooling aggregation — it collects information from neighbouring nodes and averages it.
The final prediction is a binary classification per node: is this superpixel part of a tumour or not?

---

**[Slide 14 — Training Protocol]**

Now let me explain how we trained this model.
We used the AdamW optimiser with a learning rate of 0.001 and weight decay of 0.01.
For the learning rate schedule, we used OneCycleLR with a 30% warmup phase.
The batch size was 24, and we used gradient accumulation of 2, giving an effective batch size of 48.
We also used mixed precision training — specifically AMP FP16 — to reduce memory usage.
We trained for a maximum of 50 epochs with early stopping patience of 10, but the early stopping was never triggered, meaning the model kept improving throughout.

For the loss function, we used BCEWithLogitsLoss.
Because there is a severe class imbalance — about 19 healthy superpixels for every 1 tumour superpixel — we set the positive class weight to 9.0.
This forces the model to pay more attention to tumour superpixels during training.

We ran 5-fold cross-validation at the patient level.
Each fold used 720 patients for training, 80 for validation, and 200 for testing.
After training all 5 folds, we combined their predictions using soft voting — averaging the sigmoid probabilities from all 5 models — with a decision threshold of 0.5.

I will now pass to Sakib Khan, who will present our results.

---

---

## PERSON 3 — Sakib Khan (Slides 15–19)

---

**[Slide 15 — Evaluation Metrics]**

Thank you, Rifa.
My name is Sakib Khan.
I will now present the results of our experiments.
Before showing the numbers, let me briefly explain the metrics we used.
We evaluated our model using five metrics.
The first is Dice Score — this measures the overlap between our predicted segmentation and the actual tumour region. A higher Dice means better overlap.
The second is Accuracy — the percentage of all superpixels that were correctly classified.
The third is Precision — of all the superpixels we predicted as tumour, how many were actually tumour.
The fourth is Sensitivity — of all the actual tumour superpixels, how many did we correctly find. This is also called recall.
And the fifth is Specificity — of all the healthy superpixels, how many did we correctly identify as healthy.

---

**[Slide 16 — Cross-Validation Results]**

Now let's look at the results on our 5-fold cross-validation on BraTS 2021 with 1,000 patients.

Fold 1 gave a Dice of 90.41%.
Fold 2 gave 89.58%.
Fold 3 gave 90.23%.
Fold 4 gave 89.71%.
Fold 5 gave 90.17%.

The mean Dice across all five folds is 90.02%, with a standard deviation of only 0.66%.
The very low standard deviation tells us something important — the model performs consistently across different subsets of patients.
It is not just getting lucky on one fold.

---

**[Slide 17 — Held-Out Test Results]**

Now for the most important result — the evaluation on our sealed held-out test set of 251 patients.
Remember, these patients were never used during training or fold selection.
This gives us a true measure of how well the model generalises.

The ensemble Dice score on the held-out set is **91.41%**.
The accuracy is 99.14%.
The precision is 95.52%.
The sensitivity is 87.77%.
And the specificity is 99.76%.

These numbers show that the model is not just accurate overall — it is very precise when it says something is a tumour, and it is very good at ruling out healthy tissue.
The sensitivity of 87.77% means we still correctly find nearly 88% of all actual tumour regions.

---

**[Slide 18 — Comparison with U-Net Baseline]**

To understand how good these results are, let's compare with the U-Net baseline we trained on the same hardware.
U-Net is a very well-known deep learning model for medical image segmentation.
On the same task, same hardware, same dataset, U-Net achieves a Dice of 87.84%.
Our model achieves 91.41%.
That is a 3.57 percentage point improvement in Dice.
So not only is our model much more efficient — which we will see in the next part — it is also more accurate.
This shows that using superpixel graphs and GNNs is a genuinely better approach, not just a cheaper one.

---

**[Slide 19 — Efficiency Comparison]**

Now beyond accuracy, let me show you how efficient our model is — because I believe this is equally important.
Let me show you exactly how much lighter our model is compared to U-Net.

For inference time, our full end-to-end pipeline — including superpixel construction and graph building — takes 1,732 milliseconds per patient.
U-Net takes 10,160 milliseconds.
Our method is **5.9 times faster**.
If we look at only the GNN inference part — without graph construction — it takes just 75.4 milliseconds.

For memory, our model uses only 11 megabytes of peak GPU memory.
U-Net uses 2,500 megabytes.
That is a **227 times reduction**.

For parameters, our model has 439,041 parameters.
U-Net has 69,146,113 parameters.
That is a **157 times reduction**.

For storage, our model file is 1.7 megabytes.
U-Net's model file is 264 megabytes.
Again, **157 times smaller**.

What does this mean in practice?
It means our model can run on a basic consumer GPU with just 6 gigabytes of VRAM.
It means a hospital or clinic that cannot afford a high-end computing setup can still use our system.
This is why we believe efficiency is not just a technical achievement — it is a real-world impact.
I will now hand over to Md. Minhajur Rahman, who will present our generalisation results and remaining findings.

---

---

## PERSON 4 — Md. Minhajur Rahman (Slides 20–22)

---

**[Slide 20 — BraTS 2023 Zero-Shot Generalisation]**

Thank you, Sakib.
My name is Md. Minhajur Rahman.
Sakib has shown you that our model is both accurate and highly efficient. Now let me show you how it performs on completely unseen data.
One of our four research objectives was to test whether the model can transfer to a completely different dataset without retraining.
We took our trained model — trained only on BraTS 2021 — and ran it directly on BraTS 2023.
No fine-tuning. No adaptation. Zero-shot.

The Dice score on BraTS 2023 is 89.40%.
On BraTS 2021 it was 91.41%.
So there is a small drop of about 1 percentage point.

But here is the really interesting finding — the sensitivity actually **improved**.
On BraTS 2021 the sensitivity was 87.77%.
On BraTS 2023 it jumped to 90.69%.
That is an improvement of 2.92 percentage points.

This means that on the unseen BraTS 2023 data, our model is actually better at finding tumour regions than on its own test set.
This is a very strong sign of genuine generalisation — the model has learned features that transfer across different MRI acquisition protocols.

---

**[Slide 21 — Failure Cases]**

Every model has cases where it fails, and we believe it is important to be transparent about this.
Approximately 5% of slices are not segmented well.
But more importantly, there are 3 patients in our held-out set where the model completely failed — with Dice scores essentially equal to zero.
These are patients BraTS2021_01405, BraTS2021_01366, and BraTS2021_01407.

Why did the model fail on these patients?
The root cause is absent or very faint T1CE contrast enhancement.
Remember — our entire superpixel graph is built on the T1CE channel.
If a tumour does not enhance in T1CE — which happens with certain non-enhancing tumour types — then the superpixels do not capture the tumour boundary at all.
All 15 node features lose their discriminative power.
The model simply cannot distinguish tumour from healthy tissue.

This is an honest limitation of our current design, and it directly informs our future work.

---

**[Slide 22 — Ablation Study]**

We also ran an ablation study to justify the specific architecture choices we made.
We tested four variants on Fold 0.

Our baseline — 5 layers, 256-dimensional hidden channels — achieves a Dice of 84.03% with 439,000 parameters.
Adding a 6th layer gives 84.00% Dice but increases parameters to 571,000 — slightly worse with more parameters.
Using 512-dimensional hidden channels gives a higher Dice of 88.78%, but the parameter count balloons to 1,710,000 — nearly four times more.
Replacing GraphSAGE with GAT (Graph Attention Network) gives 85.03% Dice but requires 1,184,000 parameters — also more expensive.

Our 5-layer, 256-dimensional GraphSAGE is the sweet spot.
It delivers the best efficiency-to-accuracy trade-off.
This is not a random choice — it is supported by evidence.

I will now hand over to Md. Mahamudul Hasan, who will present our conclusions, limitations, and future work.

---

---

## PERSON 5 — Md. Mahamudul Hasan (Slides 23–27)

---

**[Slide 23 — Limitations]**

Thank you, Minhajur.
My name is Md. Mahamudul Hasan.
I will close the presentation with our limitations, future work, and final conclusions.

We identified four main limitations of our current work.

First — our superpixel pipeline depends heavily on T1CE image quality.
For tumours that do not enhance in T1CE — non-enhancing gliomas, for example — the model fails completely, as we saw in the failure cases.

Second — our model only does binary segmentation.
It can tell you where the whole tumour is, but it cannot distinguish between the tumour sub-regions: the enhancing core, the necrotic core, and the surrounding oedema.
Clinical treatment planning often requires this level of detail.

Third — the graph topology is fixed at inference time.
We build the graph once and feed it through the model.
There is no mechanism to refine the graph during prediction based on the model's own confidence.

And fourth — we trained only on adult glioma cases from BraTS.
We do not know how the model performs on paediatric brain tumours or rare tumour types.
This is an important gap for clinical deployment.

---

**[Slide 24 — Future Work]**

Based on these limitations, we have five directions for future work.

First — extend the model to multi-class segmentation, separating the whole tumour into its sub-regions, as required by the BraTS evaluation hierarchy.

Second — develop dynamic graph construction that can refine superpixel boundaries during prediction, based on the model's own uncertainty estimates.

Third — validate the model on paediatric BraTS datasets to check whether it generalises to different patient populations.

Fourth — incorporate radiomics features or clinical metadata as additional node features.
For example, adding patient age or known tumour grade as extra input could improve performance on edge cases.

And fifth — explore deployment on edge hardware, such as NVIDIA Jetson or even Raspberry Pi.
If we can bring inference down to under a second on low-power hardware, this system becomes genuinely useful in low-resource clinical environments around the world.

---

**[Slide 25 — Conclusions]**

Let me now summarise what we achieved in this project.

First — our GraphSAGE-based model on superpixel graphs achieves a Dice score of 91.41% on the BraTS 2021 held-out test set.
This is 3.57 percentage points higher than the U-Net baseline trained on the same hardware.
So we are more accurate, not just lighter.

Second — the computational savings are dramatic.
Compared to U-Net: 5.9 times faster end-to-end inference, 227 times less memory, 157 times fewer parameters, and 157 times smaller model file.
All of this runs on a consumer GPU with 6 gigabytes of VRAM.

Third — zero-shot transfer to BraTS 2023 gives a Dice of 89.40%, with sensitivity actually improving by 2.92 percentage points.
The model generalises across different MRI acquisition protocols without any retraining.

And fourth — this combination of near-state-of-the-art accuracy with dramatically reduced computational cost makes this approach genuinely suitable for resource-constrained clinical environments.
We believe this is not just a research contribution — it is a step toward making AI-assisted brain tumour diagnosis accessible to more hospitals and more patients.

---

**[Slide 26 — References]**

We built on a strong body of existing work.
The BraTS benchmark datasets, the GraphSAGE algorithm, the SLIC superpixel method, and prior medical image segmentation approaches all informed our design.
The full list of references is available in our written paper.

---

**[Slide 27 — Thank You / Q&A]**

That concludes our presentation.
On behalf of all five team members — Kishor Kumar Das, Rifa Sanjida, Sakib Khan, Minhajur Rahman, and myself, Mahamudul Hasan — I want to thank our supervisor, Mr. Shamim Ahmed, for his guidance throughout this project.
We are happy to answer any questions you may have.
Thank you.

---

---

## QUICK REFERENCE — Key Numbers for Q&A

| Fact | Number |
|------|--------|
| CV mean Dice (BraTS 2021) | 90.02% ± 0.66% |
| Held-out ensemble Dice | 91.41% |
| Held-out Accuracy | 99.14% |
| Held-out Precision | 95.52% |
| Held-out Sensitivity | 87.77% |
| Held-out Specificity | 99.76% |
| U-Net Dice (baseline) | 87.84% |
| Dice improvement over U-Net | +3.57 pp |
| End-to-end inference (ours) | 1,732 ms |
| End-to-end inference (U-Net) | 10,160 ms |
| Speed improvement | 5.9× faster |
| Memory (ours) | 11 MB |
| Memory (U-Net) | 2,500 MB |
| Memory reduction | 227× |
| Parameters (ours) | 439,041 |
| Parameters (U-Net) | 69,146,113 |
| Parameter reduction | 157× |
| BraTS 2023 Dice (zero-shot) | 89.40% |
| BraTS 2023 Sensitivity | 90.69% (+2.92 pp vs BraTS 2021) |
| Total patients used | 1,251 (1,000 CV + 251 held-out) |
| Spatial compression | 2,284× (8.9M voxels → ~3,909 nodes) |
| SLIC superpixels per slice | ~46 realised (K=200 target) |
| Node feature dimensions | 15 |
| GraphSAGE layers | 5 |
| Hidden dimension | 256 |
| Complete patient failures | 3 (absent T1CE enhancement) |
| Research objectives | 4 |
| Limitations | 4 |
| Future work directions | 5 |
| Conclusions | 4 |
