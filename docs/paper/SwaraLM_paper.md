---
title: "Counting Still Wins: A Label-Free Evaluation of Melodic Representations for Carnatic Raga Retrieval"
author: "Saanchita V. Iyer"
date: "November 2026"
abstract: |
  Raga retrieval, finding other renditions of the same raga by melodic content, is a core organising task for archives of Indian art music, where expert annotation is scarce and does not scale. We present a label-free evaluation of melodic representations for query-by-example raga retrieval on the Saraga Carnatic collection (176 labelled recordings, 95 ragas, 123 evaluable queries). From tonic-normalised, symbolically denoised svara sequences we compare seven representations spanning the sparse-to-dense axis of modern retrieval: a pitch-class histogram, a first-order svara-transition matrix, BM25, an LSTM language model, a Transformer language model, contrastive fine-tuning with InfoNCE, and ColBERT-style late interaction. Count-based representations (P@1 = 0.423 and 0.431) outperform every learned alternative (best learned: 0.369 with a standard deviation of 0.016 over five held-out-artist splits), in aggregate and within the allied-raga stratum where scale content alone cannot discriminate. We further show that apparent single-run parity of contrastive embeddings is not reproduced under multi-seed evaluation; that late interaction, which removes the pooling bottleneck entirely, performs worse than the pooled model it was built from, eliminating pooling as an explanation; that BM25 retrieves at chance because every pitch position occurs in every recording, so that document frequency carries no information and every query reduces to the same twelve terms; and that symbolic denoising improves retrieval while raising language-model perplexity. A sub-study on the phrase-annotated subset reverses the picture: a term-weighted vector space model over expert phrases outperforms svara counting on the same recordings, and improves further under approximate phrase matching. We therefore state the finding as a claim about representational granularity rather than about model families: at this data scale, the unit of representation matters more than the sophistication of the model.
---

# 1 Introduction

A raga is not a scale. Two ragas can draw on exactly the same set of notes and remain, to any trained listener, entirely distinct, because what separates them is how those notes move: which phrases are characteristic, which transitions are permitted in ascent and which only in descent, where the voice lingers and where it passes through. Carnatic musicians call this movement sanchara, and it is the level at which raga identity resides [@krishna2012carnatic]. Ragas that share scale material and differ only in movement are called allied ragas, and they are the standing hard case for any computational account of raga.

This creates a specific problem for retrieval. Suppose we are given a recording and asked to find others in the same raga, with no labels available. A representation that captures which notes were sung will separate ragas that differ in note content and will fail on exactly the pairs that matter. A representation that captures movement should do better. The question is what kind of representation captures movement well enough to help, and whether learning one from data is better than counting.

The question is worth asking under label-free conditions in particular. Raga annotation is expert work and it does not scale, so the collections that most need organising are the ones least likely to be labelled. Supervised raga classification is by now well studied and reports high accuracies on well-populated raga sets [@madhusudhan2019deepsrgm; @singh2025explainable], but those results do not answer whether melodic representations learned without supervision support retrieval, which is the setting an unlabelled archive presents.

We therefore ask three questions on the Saraga Carnatic collection [@srinivasamurthy2021saraga]. First, do self-supervised learned representations outperform count-based ones for query-by-example raga retrieval at realistic data scale? Second, does any representation resolve allied ragas beyond what note content alone achieves? Third, how sensitive are these representations to the noise introduced by symbolic extraction and to the variance of a single training run?

Our answers are largely negative, and we think the negative results are the contribution. From tonic-normalised symbolic svara sequences covering 176 recordings across 95 ragas, we compare seven representations spanning the sparse-to-dense axis of modern retrieval: a pitch-class histogram, a svara-transition matrix, BM25, an LSTM language model, a Transformer language model, contrastive fine-tuning, and ColBERT-style late interaction. Count-based representations outperform every learned one, in aggregate and on the allied stratum. Apparent parity between contrastive fine-tuning and counting in a single development run is not reproduced under multi-seed evaluation. Late interaction, which exists precisely to remove the pooling bottleneck that might explain the learned models' weakness, performs worse than the pooled model it was built from, eliminating that explanation. BM25 retrieves at chance because every pitch position occurs in every recording, so document frequency is uniform and carries no information.

Each of these failures has a mechanism, and taken together they point to one conclusion. The generative objective teaches a model the grammar that ragas share rather than the movement that separates them. Graded relevance judgements make this visible: when a representation retrieves the wrong raga, it lands on an allied raga several times more often than chance would predict, so the models know the neighbourhood and not the address.

The positive result completes the argument. On the subset of recordings carrying expert phrase annotations, a term-weighted vector space model over phrases outperforms svara counting on the same recordings, and improves further when variant transcriptions of the same phrase are merged by approximate string matching. Term weighting failed over svaras and succeeds over phrases for the same reason in both cases: svaras are universal and carry no discriminative document frequency, while phrases are rare and raga-specific. The classical retrieval machinery was never the problem. It was being applied at the wrong level of granularity. We state the overall finding as a claim about representational units rather than about model families: at this data scale, the unit of representation matters more than the sophistication of the model.

Our contributions are as follows.

1. A label-free retrieval evaluation protocol for Carnatic raga retrieval, with allied-raga stratification by scale overlap, held-out-artist splits, multi-seed variance reporting, and graded relevance judgements, released for reuse along with a frozen symbolic dataset and a raga theory table.
2. A controlled seven-way comparison of count-based, lexical, generative, contrastive and late-interaction representations under that protocol, arranged so that each representation adds one capability over a simpler one and differences localise the contribution or failure of that capability.
3. Mechanistic diagnoses rather than bare results: objective misalignment supported from three independent directions, the elimination of the pooling explanation, the collapse of term weighting on a small universal alphabet, and a perplexity paradox in which denoising improves retrieval while worsening the language modelling metric.
4. A phrase-level sub-study showing that expert phrase representations do outperform counting, locating the limitation at the unit of representation rather than at the retrieval machinery.

The remainder of the paper is organised as follows. Section 2 situates the work in computational raga analysis and in retrieval. Section 3 describes the data and the symbolic extraction, including the decisions that keep the pipeline label-free. Section 4 defines the representations and Section 5 the evaluation protocol. Section 6 reports the main results and Section 7 the phrase sub-study. Section 8 discusses mechanisms and limitations, and Section 9 concludes.

# 2 Related Work

**Computational raga analysis.** The computational study of Indian art music was put on a systematic footing by the CompMusic project [@serra2014compmusic], which produced the Saraga open datasets [@srinivasamurthy2021saraga] and established core tasks including tonic identification, melody extraction and raga recognition. The classical approach to raga recognition builds on pitch-class distributions [@koduri2012raga], which is directly ancestral to our histogram baseline. @gulati2016phrase extended this line with a phrase-based vector space model, and @gulati2016tdms with time-delayed melody surfaces that capture short-range melodic movement. Our count-based baselines place this classical lineage under a retrieval protocol, and our phrase sub-study (Section 7) revisits the phrase-based vector space idea on expert annotations.

**Supervised deep raga recognition.** A substantial literature applies deep learning to raga classification. DeepSRGM [@madhusudhan2019deepsrgm] reports 97.1% accuracy on a 10-raga subset and 88.1% on the full 40 ragas of the CompMusic Carnatic Music Dataset with LSTM sequence classification, and introduces sequence ranking as a second task. PhonoNet [@chowdhuri2019phononet] applies multi-stage deep networks to Hindustani raga identification. More recent work spans explainable CNN-LSTM identification [@singh2025explainable] and the detection and clustering of ragas unseen in training [@singh2025unseen]. These systems are supervised: raga labels drive training, and evaluation is classification accuracy over 10 to 40 well-populated raga classes. Our setting differs on all three axes. No labels enter training at any stage, the task is retrieval rather than classification, and the collection is a long-tailed one of 95 ragas. Headline accuracies from this literature are therefore not comparable to the retrieval precision we report, and we say so explicitly to prevent misreading.

**Discovery and contrastive approaches.** Closest to our work, @singh2026discover adapt Generalized Category Discovery [@vaze2022gcd] to raga identification on the PIM and Saraga Hindustani datasets, combining supervised and unsupervised contrastive objectives over audio-derived embeddings and evaluating with clustering metrics. Our study addresses the Carnatic collection, operates on symbolic svara sequences, is label-free end to end, and evaluates retrieval with IR metrics under allied stratification. Their feature-extractor comparison is nonetheless relevant to ours: a CNN-LSTM trained from scratch on raga labels outperforms the large pretrained music transformers MERT [@li2024mert] and CultureMERT [@kanatas2025culturemert]. That is an independent instance of the pattern we report, in which a small, domain-fitted model beats large general-purpose representations at raga-scale data.

**Strong simple baselines and evaluation practice.** Our finding joins a long-standing observation in information retrieval that simple, well-tuned baselines are difficult to beat and that reported gains over them often fail to hold up. @armstrong2009improvements showed that a decade of reported ad hoc retrieval improvements did not accumulate because they were measured against weak baselines, and @lin2018neuralhype and @yang2019neuralhype made the same case for neural ranking models. A parallel line of work shows that single-run results overstate what is known about a neural model, and that score distributions over seeds should be reported instead [@reimers2017reporting]. Our contribution to this line is a domain in which both effects are quantified at once, against extraction noise and against run-to-run variance.

**Positioning.** To our knowledge, no prior work evaluates label-free learned representations against count-based baselines for query-by-example raga retrieval on Saraga Carnatic with allied-raga stratification. We frame the work not as a new method but as a controlled evaluation with mechanistic diagnosis, in the tradition of evaluation-centred music information retrieval [@downie2008mirex], and we release the protocol for reuse.

# 3 Data and Symbolic Representation

## 3.1 The Saraga Carnatic collection

We use the Carnatic collection of the Saraga dataset [@srinivasamurthy2021saraga; @bozkurt2018saraga], one of the largest open, annotated corpora of Carnatic music. Version 1.5 of the collection contains 249 concert tracks. Our pipeline requires two annotations, a per-recording tonic (the performer's sa) and a predominant-melody f0 contour, and 197 tracks carry both. These 197 recordings are our working collection. Editorial metadata links each recording to its raga, lead artist and concert.

A census of the working collection reveals a pronounced long tail. The raga field takes 97 distinct values, two of which are not ragas and define the recordings we exclude. Thirteen recordings carry no raga label in the collection metadata, and we did not attempt to supply one: assigning labels by ear would introduce an annotation of different provenance from the rest of the collection, and the recordings are too few to justify the inconsistency. Eight further recordings are labelled only as Ragamalika, the generic label for compositions that move through several ragas, which admits no single ground truth for a retrieval task defined on raga identity. Excluding both groups leaves 176 labelled recordings across 95 ragas and 17 lead artists. Of the 95 ragas, 53 appear exactly once, and only 18 have at least three recordings by at least two distinct lead artists. The 123 recordings whose raga occurs at least twice, covering 42 ragas, serve as evaluable retrieval queries (Section 5.1). The remaining 53 recordings, each the only representative of its raga, stay in the collection as distractors.

The long-tailed, small-scale character of this collection is not a limitation to be engineered around but the realistic condition of the task. Expert raga annotation is costly, and archives of underrepresented traditions are precisely where label-free methods are needed.

## 3.2 From f0 to svara sequences

Each recording is converted to a symbolic sequence of svara-position tokens in three steps. (Svara is also transliterated swara, the spelling used in the project name and in the released code.)

**Tonic normalisation.** Every f0 value in hertz is converted to cents relative to the recording's annotated tonic, $c = 1200 \log_2(f_0 / f_{\mathrm{tonic}})$. This removes the performer-specific tonic, the dominant nuisance variable in cross-recording melodic comparison, by construction. Unvoiced frames ($f_0 = 0$) are mapped to a silence token.

**Quantisation.** Cents values are folded to a single octave and assigned to the nearest of 12 pitch positions spaced 100 cents apart. Carnatic theory names 16 svarasthanas, but they occupy only 12 distinct pitch positions: four pairs are enharmonic, sharing a frequency while differing in name and musical function. The pairs are R2/G1 (about 200 cents), R3/G2 (about 300), D2/N1 (about 900) and D3/N2 (about 1000). Because frame-level pitch reports only a frequency, it cannot recover which member of a pair was intended, and we therefore represent each shared position with a single token.

The consequence is concrete. Chalanata (S R3 G3 M1 P D3 N3) folds to the token sequence S G2 G3 M1 P N2 N3: its R3 is absorbed into the G2 token and its D3 into the N2 token, so the representation can no longer register that R3 and D3 were the svaras sung. Kharaharapriya (S R2 G2 M1 P D2 N2), which really does contain G2 and N2, occupies overlapping positions. The same collision recurs independently of the fourth degree: Rasikapriya and Hemavathi are the M2 counterparts of Chalanata and Kharaharapriya and collide identically at the 300-cent and 1000-cent positions.

This folding is not a mere labelling convenience. It discards real information. In performance, R3 and G2 are not the same note. They differ in gamaka (ornamental oscillation), in stress, and in the way they are approached and left. The raga is defined by svarasthana together with movement, not by frequency, and a trained listener distinguishes R3 from G2 by exactly this movement. The enharmonic partners we fold are not theoretical curiosities absent from the data: shuddha gandhara (G1) occurs in the corpus, for example in Varali, Kanakangi and Ganamurthi, and shuddha nishada (N1) occurs in Kanakangi, so folding erases distinctions that are present in the recordings. We accept this loss deliberately, because recovering it would require either the raga label, which the task forbids (Section 3.3), or the movement information that frame-level pitch does not carry. It is, in miniature, the premise of the paper: at the colliding positions note content is identical and only movement separates the svaras, which is precisely the information a pitch-position representation cannot see. The per-note gamaka descriptors we retain (below) are what a future representation could use to recover it.

**Note segmentation.** The predominant-melody pitch track has a frame hop of 4.44 ms. Smoothing is applied to the f0 series in hertz before conversion to cents; because the cents mapping is monotonic in f0, median filtering commutes with it and the two orders are equivalent. The filter operates on the voiced frames alone, with unvoiced frames removed beforehand, so a window may in principle span a short silence, a minor effect we note for completeness. The frame-level stream is median-filtered (window of 21 frames, about 93 ms) and then quantised. Runs shorter than a minimum dwell (10 frames, about 44 ms) are discarded, and consecutive identical tokens are collapsed to single note events, with silences retained as phrase boundaries. For each note we additionally record a gamaka profile, the within-note standard deviation of pitch in cents and the net drift, preserving ornamentation as explicit features rather than discarding it. These features are released with the data but are not used by the representations evaluated here (Section 8.8).

## 3.3 Denoising without label leakage

An initial extraction with light smoothing (window of 5 frames, minimum dwell of 3), which we call v1, produced sequences contaminated by gamaka spillover: oscillations around a held svara were emitted as spurious neighbouring tokens. In renditions of raga Vasanta, for example, M2 and G2 tokens appeared although the raga contains neither. Inspection by the author, a trained Carnatic vocalist, identified these as artefacts of ornamentation passing through adjacent quantisation bins rather than sung notes.

Two remedies are possible, and the choice between them is consequential. A raga-aware remedy would correct extracted tokens against each raga's theoretical scale. In a Vasanta recording, the spurious G2 and M2 tokens could simply be deleted or snapped to neighbouring scale tones, since Vasanta's grammar admits neither. This removes the artefacts cleanly, but it is inadmissible for two reasons. First, the correction requires knowing that the recording is Vasanta in order to decide that G2 and M2 do not belong. It injects the raga label into the input of models whose task is to recover raga structure. This is label leakage: the representation would encode the answer it is meant to discover, and the corrected data could not be produced at all for a query recording whose raga is unknown, which is precisely the retrieval setting. Second, the oscillations being corrected are gamaka, the movement that distinguishes ragas sharing a scale (Section 3.2), and snapping them to scale tones erases the information the task most depends on.

We therefore adopt a strictly raga-blind remedy, the v2 extraction: stronger median smoothing and a longer minimum dwell, both functions of the signal alone. Raga theory is used only outside the extraction path. At extraction time, a small table of directional arohana (ascent) and avarohana (descent) for a subset of ragas supports two post hoc health checks: the fraction of frame energy falling on a raga's theoretical notes, and the fraction of melodic steps consistent with the raga's directional grammar. The table stores ascent and descent separately rather than as a flat set, because a svara legal in descent may be illegal in ascent. In Bilahari, Ma and Ni appear only in avarohana, and a flat set would erase an asymmetry that is itself part of sanchara. A larger theory table covering all 95 evaluated ragas, supplied by the author and carrying a confidence flag for each entry, is used for evaluation only: to compute the scale-overlap statistic that defines the allied-raga stratum (Section 5.3) and the graded relevance judgements for nDCG (Section 5.2). Neither table ever modifies an extracted sequence, and both are released with the paper.

The denoising choice proves consequential for the study's conclusions. As Section 6.3 shows, representations that model the order of svaras are roughly twice as sensitive to this symbolic noise as the order-free histogram, and language-model perplexity paradoxically increases under denoising because the removed jitter was the most predictable part of the noisy sequences.

## 3.4 Dataset freezing and provenance

Symbolic sequences were extracted from a public mirror of the Saraga Carnatic collection and subsequently cross-validated against the canonical release, Saraga Carnatic v1.5 [@bozkurt2018saraga], accessed through the mirdata loader with checksum verification [@bittner2019mirdata]. That cross-validation confirmed the annotation coverage reported in Section 3.1 (197 of 249 tracks carry both tonic and predominant-melody pitch; vocal-only pitch covers 56 tracks and was therefore not used) and established the 4.44 ms frame hop quoted above. The phrase sub-study (Section 7) reads its annotations directly from the same v1.5 release. The collection is distributed under the CC BY-NC-SA 4.0 licence.

Because extraction parameters materially affect downstream results (Section 6.3), the final v2 extraction was frozen to a versioned file, and every result reported in this paper is computed on this single frozen version unless explicitly marked as v1. We verified internal consistency by checking that stratified metrics aggregate exactly to overall metrics across the shared query set. We release the frozen symbolic dataset, the census and the raga theory tables alongside the code.

# 4 Representations

All representations map a recording's svara sequence to a vector or, for BM25 and late interaction, directly to a scoring function. Retrieval then ranks by cosine similarity over L2-normalised vectors unless stated otherwise (Section 5.2), so the representation is the only varying component. We evaluate two count-based representations, one lexical scoring function and four learned representations, deliberately spanning the sparse-versus-dense axis familiar from text retrieval, transplanted to a symbolic musical vocabulary of twelve tokens.

## 4.1 Count-based representations

**Pitch-class histogram.** A 12-dimensional vector of normalised svara frequencies. This is the melodic analogue of bag-of-words and instantiates the classical pitch-distribution approach to raga recognition [@koduri2012raga]. It encodes which svaras a recording dwells on and how much, and nothing about order. By design, it is the representation that allied ragas should defeat.

**Svara-transition matrix.** A $12 \times 12$ first-order Markov matrix of row-normalised transition probabilities, flattened to 144 dimensions. This is the minimal representation of movement: which svara follows which. It captures a first shadow of sanchara while remaining count-based and free of training. Its weaknesses are anticipated in its construction. Its 144 parameters are estimated from a single recording's note runs and are correspondingly noisier than the histogram's twelve, and the matrix aggregates ascending and descending contexts, discarding the directional asymmetry between arohana and avarohana that partly defines raga grammar.

**BM25.** To connect our count baselines to the standard lexical scoring function of text retrieval, we additionally score retrieval with Okapi BM25 [@robertson2009bm25] with $k_1 = 1.2$ and $b = 0.75$, treating each recording's svara sequence as a document and the query recording's distinct svaras as query terms. Two caveats make BM25 a reference point rather than a natural baseline here. First, BM25 assumes short queries against longer documents, whereas our task is symmetric, recording against recording, and cosine over normalised counts is the standard choice in that regime [@manning2008ir]. Second, with a twelve-token vocabulary, the presence-based machinery of BM25 has nothing to work with: every one of the twelve pitch positions occurs in all 176 recordings. Document frequency is therefore the same for every term, so inverse document frequency is a constant that weights nothing, and every query presents the same twelve terms, so the ranking BM25 produces does not depend on the query at all. The resulting retrieval is at chance (P@1 = 0.024; Section 6.5). We report this not as a failure of BM25 but as a property of the domain. Continuous gamaka traverses the full pitch space, so presence is universal and only the proportions among universally present symbols carry raga information, which is exactly the signal that presence-based lexical matching discards. Because the degeneracy is structural, $k_1$ and $b$ cannot repair it and we do not sweep them.

## 4.2 Learned representations

All learned models are trained exclusively on the svara sequences, with no raga labels at any stage, under held-out-artist validation (Section 5.4). Architecturally the first three are bi-encoders: a single encoder embeds every recording, and retrieval is nearest-neighbour search in the shared space.

**LSTM language model.** A deliberately small recurrent model [@hochreiter1997lstm] with a single layer, 64 hidden units, 32-dimensional token embeddings and dropout of 0.3, trained on next-svara prediction over overlapping 64-token windows with a stride of 32. Windowing converts 176 recordings into many thousands of training examples. A recording's embedding is the mean of its hidden states. The hypothesis this model tests is that predicting melodic continuations forces the model to internalise movement grammar, which mean-pooled states should then expose for retrieval.

**Transformer language model.** A two-layer causal Transformer [@vaswani2017attention] with 4 attention heads and matched embedding width, trained on the identical objective, data and validation protocol. Its purpose is a controlled architecture comparison. If the LSTM's retrieval weakness reflects the objective rather than the architecture, the Transformer should be weak in the same way, and it is (Section 6.1).

**Contrastive fine-tuning.** This is the targeted response to the objective misalignment diagnosed in Section 8.2. The LSTM encoder, warm-started from its language-model weights and extended with a projection head, is fine-tuned with the InfoNCE loss [@oord2018cpc] at temperature 0.1, where positives are two windows sampled from the same recording and negatives are in-batch windows from other recordings. Supervision remains label-free: the signal is recording identity, which costs nothing. The hypothesis is that recording discrimination transfers to raga discrimination, since two windows of one recording share a raga's movement grammar. A known structural limitation is stated up front. In-batch negatives include same-raga recordings, so on a collection where the evaluable ragas are exactly the multi-recording ones, the objective actively separates some pairs the evaluation needs together. This is the false-negative problem of contrastive learning without labels [@chuang2020debiased]. It bounds what label-free contrastive learning can achieve here, and we treat it as part of the finding rather than an implementation defect.

**Late interaction.** Every learned representation above compresses a sequence into a single vector by mean pooling, which is an obvious candidate explanation for their weakness: pooling may destroy the fine-grained structure that distinguishes ragas. We test that explanation directly with late interaction in the style of ColBERT [@khattab2020colbert]. The encoder emits one embedding per token, with no pooling anywhere, and a pair is scored by MaxSim, in which each query token contributes its best match anywhere in the document: $\mathrm{score}(Q, D) = \sum_{q \in Q} \max_{d \in D} \cos(E_q, E_d)$. We evaluate two conditions: warm-started from the trained LSTM with a fresh token projection, and fully trained with a late-interaction contrastive objective (positives are two 64-token segments of the same recording, with in-batch negatives, MaxSim scoring and held-out-artist validation). Two adaptations must be stated. ColBERT assumes an asymmetric setting, a short query against longer documents, whereas ours is symmetric, so we sample four fixed-length query segments per query recording and average their MaxSim scores. And ColBERT's power derives from contextual embeddings over a large vocabulary, whereas ours has twelve tokens, a plausible failure mode to which Section 8.3 returns.

## 4.3 What the set of representations spans

The seven representations form a controlled progression: from pure content (the histogram), to lexical scoring over content (BM25), to first-order movement (transitions), to learned sequential context under a generative objective (LSTM, Transformer), to learned context under a discriminative objective (contrastive), to discriminative learning without pooling (late interaction). Each step changes one thing, so differences in retrieval quality localise the contribution or failure of that thing. This design is what licenses the paper's mechanistic claims. When the generative models trail the histogram, the gap is attributable to the objective. When the contrastive model fails to close it, the residual is attributable to data scale and negative contamination rather than to architecture. And when late interaction, the one design that removes pooling entirely, performs worse still, the pooling explanation is eliminated rather than merely weakened.

# 5 Evaluation Protocol

## 5.1 Task and query set

We evaluate all representations on query-by-example raga retrieval. Each recording serves as a query against the remaining collection, and a retrieved recording is relevant if it shares the query's raga label. This framing evaluates the representation directly. No classifier is fit and no labels enter training, so differences between representations are not confounded by a supervised head.

Only recordings whose raga has at least two recordings in the collection can serve as queries, since otherwise no relevant item exists. This yields 123 evaluable queries among the 176 labelled recordings. The single-recording ragas stay in the collection as distractors, which keeps the retrieval pool realistic: a query must find its same-raga counterparts among 175 candidates spanning 95 ragas, not within a curated closed set.

## 5.2 Metrics

Similarity is cosine over L2-normalised vectors for every representation, count-based and learned alike, so the similarity function is held constant and only the representation varies. The two exceptions are BM25, which is a scoring function rather than a vector space, and late interaction, which scores by MaxSim over per-token embeddings (Section 4.2). We report precision at $k$ for $k \in \{1, 3, 5\}$, mean average precision (MAP) and mean reciprocal rank (MRR), all standard [@manning2008ir]. P@1 is the headline metric. Its chance level on this collection, the probability that a randomly chosen candidate shares the query's raga, averages 0.015 over the 123 queries, so the observed range of 0.28 to 0.43 reflects substantial signal in every representation except BM25.

**Graded relevance and nDCG.** Binary relevance treats every incorrect retrieval identically, but musically it should not. Returning a recording of an allied raga, one sharing most of the query raga's scale, is a more defensible error than returning an unrelated raga. We therefore also report normalised discounted cumulative gain [@jarvelin2002cumulated] under two relevance schemes. The first is conventional and binary, with a gain of 1 for a same-raga retrieval and 0 otherwise. The second is graded: a gain of 2 for the same raga, 1 for an allied raga (scale overlap of at least 0.8 with the query raga, computed from the theory table as in Section 5.3), and 0 otherwise. The quantity of interest is the difference between the two for each representation. A representation whose score rises substantially under graded relevance is making musically coherent errors, landing in the right scale neighbourhood even when it fails to identify the exact raga, while one that gains little is failing arbitrarily. No binary metric can distinguish these two failure modes.

## 5.3 Allied-raga stratification

Aggregate metrics could in principle hide the phenomenon of interest. Any representation that captures pitch-class statistics will separate ragas that differ in note content, and the ragas for which movement matters are those with a close scale neighbour. We therefore stratify by scale overlap. For each pair of ragas we compute the Jaccard overlap of their theoretical note sets (the union of arohana and avarohana from the theory table, after enharmonic folding), and assign each query raga to the high-overlap stratum if its overlap with some other raga in the collection is at least 0.8.

This yields 119 high-overlap queries and 4 low-overlap queries. The imbalance is itself informative: under this criterion almost every multi-recording raga in the collection has a close scale neighbour, and the low-overlap stratum consists of the four recordings of a single raga, Shanmukhapriya. The allied stratum therefore tracks the aggregate closely by construction, and its role in this study is to confirm that no representation's aggregate score is carried by easy queries, rather than to isolate a small hard subset. The low-overlap stratum is reported for completeness and excluded from all conclusions.

We complement the stratified metrics with an analysis of errors: for a representation's incorrect top-1 retrievals, how often the retrieved raga is a scale neighbour of the query raga, compared with the rate expected if errors were drawn at random from the collection (Sections 6.4 and 6.6).

## 5.4 Held-out-artist splits and multi-seed robustness

All learned models are trained with whole-artist holdout: approximately 20% of lead artists (three of seventeen) are excluded from training, and validation loss is computed only on their recordings. This is the protocol's principal measure against shortcuts. A model can reduce training loss by memorising performer-specific regularities, and under artist holdout such shortcuts fail to transfer and surface as validation degradation rather than silent inflation. Two further measures close the remaining shortcut channels: the input is purely symbolic, leaving no timbre or absolute pitch for artist identity to inhabit, and tonic normalisation removes the register cue at the source.

A single artist split is one draw from a wide distribution. The distribution of recordings over the 17 lead artists is heavily skewed, with one artist accounting for 40 of the 176 recordings and the median artist for 8, so which artists are held out changes the difficulty of the validation set considerably. For the two models at the centre of the comparison, the LSTM language model and its contrastive fine-tuning, we therefore report P@1 as a mean with standard deviation over five split seeds, each seed drawing a new artist split and retraining both stages from initialisation. Count-based representations involve no training and are reported as deterministic values. This protocol proved decisive. The contrastive model's development run reached P@1 = 0.423, apparent parity with the count baseline, and no run of the five-seed protocol reproduced it (0.366 with a standard deviation of 0.025; Section 6.2). We report both numbers because the discrepancy is itself a finding about evaluation practice at this data scale.

Where a result rests on a small query set, specifically the phrase sub-study of Section 7 with its 60 queries, we additionally report 95% confidence intervals obtained by bootstrap resampling of the query set with 1,000 resamples [@efron1993bootstrap]. Multi-seed variance and bootstrap intervals answer different questions. The former captures sensitivity to the training run and the artists held out, the latter sensitivity to which queries happen to be in the evaluation set. Count-based representations, which involve no training, are affected only by the second.

## 5.5 Provenance and internal consistency

All results in this paper derive from a single frozen version of the symbolic dataset (Section 3.4), except the v1 figures of Section 6.3, which are marked as such. As an internal consistency check, we verify that stratified metrics aggregate exactly to overall metrics across the shared query set: the 119 and 4 queries of the two strata must reproduce each aggregate P@1 as their weighted average. This check detected one data-provenance error during development, in which the stratified analysis and the first language-model runs were executed against the stale v1 extraction. It surfaced as a failed aggregation identity and an anomalous loss curve, and it motivated the freezing protocol. The v1 results it left behind are the basis of the sensitivity analysis in Section 6.3.

# 6 Results

## 6.1 Main comparison

Table 1 reports retrieval performance for all seven representations on the frozen dataset (176 labelled recordings, 95 ragas, 123 evaluable queries).

| Representation | P@1 | P@3 | P@5 | MAP | MRR |
|:---|:---|:---|:---|:---|:---|
| Svara-transition matrix | 0.431 | 0.320 | 0.237 | 0.418 | 0.547 |
| Pitch-class histogram | 0.423 | 0.290 | 0.233 | 0.423 | 0.519 |
| LSTM language model | 0.369 (0.016) | 0.282 | 0.229 | 0.405 | 0.501 |
| Contrastive fine-tuning | 0.366 (0.025) | 0.298 | 0.237 | 0.415 | 0.524 |
| Late interaction, warm-started | 0.317 | 0.263 | 0.218 | 0.352 | 0.441 |
| Late interaction, trained | 0.301 | 0.236 | 0.189 | 0.321 | 0.417 |
| Transformer language model | 0.276 | 0.201 | 0.164 | 0.276 | 0.384 |
| BM25 | 0.024 | 0.022 | 0.023 | 0.046 | 0.074 |

: Query-by-example raga retrieval on the full collection. For the LSTM and contrastive rows, P@1 is the mean over five held-out-artist splits with the standard deviation in parentheses. The multi-seed harness recorded P@1 only, so the remaining columns of those two rows come from the development runs whose embeddings we release; those runs have P@1 of 0.398 and 0.423 and so flatter both models relative to their multi-seed means. The Transformer and late-interaction rows are single development runs. Count-based rows and BM25 are deterministic.

The ordering is the paper's central result. Both count-based representations outperform every learned representation we trained. Within the learned group nothing we added helped: a discriminative objective did not improve on the generative one, removing the pooling bottleneck made retrieval worse, and a more expressive architecture did worst of all.

Two entries deserve immediate comment. BM25 retrieves at chance, for reasons given in Section 6.5. Late interaction is reported in both a warm-started and a fully trained condition because the warm start alone would have left the result open to the objection that its token projection was untrained. Training it properly, with validation loss falling from 17.1 to 1.9 under held-out-artist validation and early stopping at epoch 27, moved retrieval from 0.317 to 0.301, that is, slightly downward. The optimisation succeeded and the transfer did not.

## 6.2 Single-run parity is not reproduced under multi-seed evaluation

In the development run, on the artist split of seed 42, contrastive fine-tuning reached P@1 = 0.423, exactly matching the pitch-class histogram. Read on its own, that number supports a claim of parity between label-free contrastive learning and count-based retrieval.

Repeating the experiment across five artist splits, retraining both the language model and the contrastive stage from initialisation each time, gives 0.366 with a standard deviation of 0.025. The individual seed values are 0.350, 0.350, 0.350, 0.374 and 0.407, so the development figure of 0.423 lies above every one of them. The histogram sits at 0.423 and the transition matrix at 0.431, both more than two standard deviations above the contrastive mean. The corresponding figures for the LSTM language model are 0.369 with a standard deviation of 0.016, with seed values of 0.358, 0.358, 0.358, 0.382 and 0.390.

Two caveats apply to this comparison, and both strengthen rather than weaken its practical message. First, the multi-seed harness used a more conservative contrastive configuration than the development run: batches of 16 recordings rather than 32, a learning rate of $3 \times 10^{-4}$ rather than $10^{-3}$, and a longer schedule with more patience. The gap between the development figure and the harness value for the same seed (0.423 against 0.350) therefore mixes a configuration effect with run-to-run variation and cannot be attributed to the choice of split alone. Second, training is not exactly reproducible in our environment even with configuration and seed held fixed. The LSTM language model, whose configuration is identical in the development script and the harness, gave 0.358 in its original development run and in the harness on the same seed, but 0.398 in another run of the development script, which is the run whose embeddings we release and which Tables 3 and 4 use. A single model at this scale thus moves by several points of P@1 between nominally identical runs.

We report this at length because the methodological point generalises beyond this dataset [@reimers2017reporting]. One run of one configuration suggested parity with counting, and no run of a five-seed protocol reproduced it. Had we stopped at the development run, we would have published a parity claim that the data does not support.

## 6.3 Sensitivity to symbolic noise, and the perplexity paradox

The two extraction versions described in Section 3.3 give a natural sensitivity test. Table 2 compares P@1 under the lightly smoothed v1 extraction and the raga-blind denoised v2 extraction.

| Representation | P@1 on v1 | P@1 on v2 | Ratio |
|:---|:---|:---|:---|
| Pitch-class histogram | 0.317 | 0.423 | 1.3 |
| Svara-transition matrix | 0.220 | 0.431 | 2.0 |
| LSTM language model | 0.179 | 0.369 | 2.1 |
| Transformer language model | 0.154 | 0.276 | 1.8 |

: Retrieval before and after raga-blind denoising. The v1 values for the two count-based rows are reconstructed from the stratified v1 run, which recorded P@1 of 0.303 and 0.193 on the 119 high-overlap queries and 0.75 and 1.00 on the 4 low-overlap queries; the aggregate follows by the weighted-average identity of Section 5.5. The v1 values for the learned models are single runs. The v2 value for the LSTM is its multi-seed mean.

Every representation improves under denoising, but not equally. The histogram, which ignores order, gains about a third. The three representations that model the order of svaras, including the purely count-based transition matrix, roughly double. The mechanism is simple. A spurious token emitted by gamaka spillover adds one count to a histogram, where it is diluted by the mass of correctly extracted notes. In a sequence it corrupts two transitions, the one into the spurious token and the one out of it, and a sequence model trained on next-token prediction spends its capacity modelling those spurious transitions. Sensitivity to extraction noise is therefore a property of modelling order, not of learning as such: the transition matrix, which involves no training, behaves here like the learned sequence models. One consequence is that on the noisy v1 extraction the transition matrix fell well below the histogram, and it only became competitive once the extraction was cleaned.

The same comparison has a counterintuitive corollary. Denoising the data raised language-model perplexity, from about 3.9 on v1 to a little above 7 on v2. Cleaner data made the prediction task harder, because the jitter removed by denoising was the most predictable content in the sequences: an oscillation between two adjacent quantisation bins is trivially anticipated, whereas the melodic movement that remains after smoothing is less certain. A model selected by perplexity alone would prefer the noisier corpus. We note this as a caution for symbolic music language modelling wherever perplexity is read as a proxy for musical competence.

## 6.4 Allied ragas

Table 3 reports P@1 separately for the high-overlap stratum, the 119 queries whose raga has a scale overlap of at least 0.8 with another raga in the collection, alongside the aggregate and the 4 low-overlap queries.

| Representation | All (123) | Allied (119) | Low overlap (4) |
|:---|:---|:---|:---|
| Pitch-class histogram | 0.423 | 0.412 | 0.75 |
| Svara-transition matrix | 0.431 | 0.412 | 1.00 |
| Contrastive fine-tuning (development run) | 0.423 | 0.403 | 1.00 |
| LSTM language model (development run) | 0.398 | 0.378 | 1.00 |
| Transformer language model | 0.276 | 0.269 | 0.50 |

: Retrieval by scale-overlap stratum. The contrastive and LSTM rows use the released development-run embeddings, which lie above their multi-seed means (Section 6.2).

No learned representation resolves allied ragas better than counting. The contrastive development run comes closest, at 0.403 against 0.412, and that is the run which already overstates the model relative to its multi-seed mean. As Section 5.3 anticipated, the allied figures sit within about two points of the aggregate for every representation, so the aggregate ordering of Table 1 is not an artefact of easy queries. The four low-overlap queries are too few to interpret.

The errors themselves are musically patterned. Averaged over queries, 19% of the wrong-raga candidates in the collection have a scale overlap of at least 0.75 with the query raga. Among the incorrect top-1 retrievals of the three learned representations for which we release embeddings, the proportion is 47% to 50%. In the confusion analysis of the count-based representations, carried out on the v1 extraction, the most frequent confusions were between ragas with identical theoretical note sets: Harikambhoji and Kambhoji, Kamas and Kambhoji, and Kharaharapriya and Ritigaula. The systems fail disproportionately on the pairs that theory predicts to be difficult, which is evidence that they are behaving musically rather than arbitrarily.

A qualitative example makes the picture concrete. For a Kamas query, the contrastive representation returns three same-raga recordings in its top five and no Kambhoji recording, whereas the pitch-class histogram returns two same-raga recordings and places a Kambhoji recording at rank five. The contrastive encoder is not useless on allied material. It is simply not better on aggregate than counting.

## 6.5 BM25 and the collapse of term weighting

BM25 retrieves at P@1 = 0.024, which is chance for this collection. The cause is structural rather than a matter of parameter choice. Every one of the twelve pitch positions appears in all 176 recordings. Two things follow. Document frequency is identical for every term, so inverse document frequency is a constant and no term is weighted above any other. In the classical formula that constant is negative for a term present in every document, and implementations differ in how they floor it, but whatever its value it cannot discriminate. And since every recording contains all twelve symbols, every query, represented by its distinct svaras, is the same query. BM25 therefore assigns each document a score that does not depend on which recording is asked about, and returns one fixed ranking of the collection for all 123 queries. A query-independent ranking retrieves at chance by construction: the observed P@1 of 0.024 is 3 correct out of 123, consistent with a single fixed ranking whose top recording shares its raga with three of the queries.

The reason every position appears everywhere is musical. Continuous gamaka traverses the pitch space between the notes a raga admits, and after quantisation these transits deposit at least one note event in every bin over the course of a concert recording. Presence therefore carries no information. Only the relative proportions among universally present symbols do, which is precisely what presence-based lexical matching discards. The histogram, built from those proportions, retrieves at 0.423 from the same underlying counts. A variant of BM25 that weighted query terms by their frequency would reintroduce the proportions and move the scoring function towards the histogram comparison; we did not evaluate it.

We report this as a domain property rather than a deficiency of BM25. It is a concrete instance of a general caution: retrieval machinery developed for natural language assumes a large, heavy-tailed vocabulary, and transplanting it to a small symbolic alphabet requires checking that assumption first.

## 6.6 Graded relevance: the errors are musical

Table 4 reports nDCG at rank 1 under binary relevance (same raga scores 1) and under graded relevance (same raga scores 2, allied raga scores 1).

| Representation | Binary | Graded | Gain |
|:---|:---|:---|:---|
| Svara-transition matrix | 0.431 | 0.537 | 0.106 |
| Pitch-class histogram | 0.423 | 0.528 | 0.105 |
| Contrastive fine-tuning (development run) | 0.423 | 0.508 | 0.085 |
| LSTM language model (development run) | 0.398 | 0.500 | 0.102 |
| Transformer language model | 0.276 | 0.394 | 0.118 |

: Binary and graded nDCG at rank 1. Binary nDCG at rank 1 equals P@1.

Every representation gains between 0.085 and 0.118 under graded relevance. At rank 1 the gain has a direct reading: it is half the fraction of queries whose top retrieval is an allied raga rather than the correct one. For every representation in the table, between 30% and 37% of incorrect top-1 retrievals fall on an allied raga. The rate expected if errors were drawn at random from the wrong-raga candidates is 9%. Errors land in the query raga's scale neighbourhood three to four times more often than chance, and this holds for the Transformer, by some margin the weakest representation in Table 1, as much as for the strongest.

This is one of three lines of evidence for the objective misalignment we diagnose in Section 8.2. The learned models did acquire real knowledge of Carnatic melody, enough to place a disproportionate share of their errors in the correct region of the raga space, but not the finer discrimination that separates ragas within that region. They learned what ragas share and not what distinguishes them.

## 6.7 Summary

At this data scale, on this collection, and under a label-free protocol: count-based svara representations outperform generative sequence models, contrastive fine-tuning and late interaction, both overall and on allied ragas; a single run materially overstates the learned models; representations that model svara order are about twice as sensitive to symbolic extraction noise as the order-free histogram; perplexity moves opposite to retrieval quality under denoising; lexical term weighting collapses entirely on a twelve-symbol vocabulary; and all representations err towards musically adjacent ragas far more often than chance. Section 7 reports a sub-study on the phrase-annotated subset in which a phrase-level representation does overtake counting.

# 7 Phrase-Level Representation: A Sub-Study

## 7.1 Motivation

The representations compared in Section 6 operate at the level of individual svaras: histograms, transitions and sequence models over a twelve-symbol alphabet. Our central negative result, that count-based svara representations are not beaten by learned svara-sequence models at this scale, leaves open a different question: is the svara the right unit at all? Practitioners often describe raga identity not through individual notes but through characteristic phrases, short melodic gestures whose presence and shape signal a raga more sharply than its note content [@krishna2012carnatic]. The Saraga Carnatic collection includes expert phrase annotations for a subset of recordings, transcribed as solfege strings such as `ndnsndpm` and `srgrsnsrgrs`. We use them to test whether a phrase-level representation captures raga identity that svara-level counting misses.

## 7.2 Data and protocol

Phrase annotations are available for 100 of the 249 tracks. Excluding Ragamalika and unlabelled items leaves 94 recordings across 58 ragas, of which 22 ragas have at least two recordings, yielding 60 evaluable queries. The annotations comprise 1,474 phrase instances and 535 distinct phrase strings. This is a smaller and distinct evaluation set from the main study (123 queries over 42 ragas), so all numbers below are reported and interpreted only relative to one another, never against the main-study figures. We retain the retrieval protocol of Section 5 (query by example, cosine similarity, P@1, MAP and MRR) and, given the small query set, report 95% confidence intervals from 1,000 bootstrap resamples over queries.

## 7.3 Representations

We compare three representations built from the same phrase-annotated data.

1. **Svara counts.** A histogram over the solfege symbols of a recording's phrase transcriptions. This is the bag-of-notes idea of Section 4.1 restricted to this subset, as a same-data baseline.
2. **Phrase VSM (exact).** Each recording is a bag of its phrase strings. We build a TF-IDF vector space model [@salton1988term] over the phrase vocabulary of 535 distinct strings and retrieve by cosine similarity. This instantiates the phrase-based vector space approach of @gulati2016phrase under our protocol, with expert annotations in place of automatically discovered phrases.
3. **Phrase VSM (approximate).** Exact string matching treats transcription variants of one gesture, for example `ndnsndpm` and `ndnsndppm`, as distinct terms. We first cluster the phrase vocabulary by normalised edit-distance similarity, merging variants above a threshold $\tau$ into a canonical term, and then build the TF-IDF model over canonical phrases. This is a record-linkage step, the same class of approximate string matching used for entity resolution in data systems, applied to melodic transcriptions.

## 7.4 Results

A recurrence probe motivates the approach. Within a raga, the Jaccard overlap of two recordings' phrase sets averages 0.018. Across ragas it rounds to 0.000, and the ratio of the two is 57. Phrases essentially never recur across different ragas, so their presence is strongly discriminative. This is the opposite of the svara vocabulary, where every symbol occurs in every recording and document frequency is uninformative (Section 6.5). Over phrases, inverse document frequency is meaningful, because phrases are rare and raga-specific.

Table 5 reports retrieval with 95% bootstrap confidence intervals.

| Representation | Terms | P@1 | MAP | MRR |
|:---|:---|:---|:---|:---|
| Svara counts (same data) | n/a | 0.150 [0.07, 0.25] | 0.212 [0.15, 0.29] | 0.276 [0.20, 0.37] |
| Phrase VSM (exact) | 535 | 0.333 [0.22, 0.45] | 0.291 [0.20, 0.38] | 0.391 [0.29, 0.50] |
| Phrase VSM (approximate) | 459 | 0.450 [0.33, 0.58] | 0.406 [0.30, 0.52] | 0.519 [0.41, 0.63] |

: Retrieval on the phrase-annotated subset (60 queries, 22 ragas), with 95% bootstrap confidence intervals. The approximate model uses a merge threshold of $\tau = 0.85$.

The clearest finding is that the approximate phrase model outperforms svara counting on the same data: its confidence intervals are disjoint from those of svara counts on all three metrics. The two intermediate steps are directional. Exact phrase matching roughly doubles svara-count P@1 (0.333 against 0.150), and approximate matching improves on exact matching by a further 0.117, but in both cases the intervals of adjacent rows overlap, so with 60 queries neither step can be separated from its neighbour on interval evidence alone.

The gain from approximate matching is supported instead by its behaviour under the merge threshold. Table 6 shows a stable optimum where meaningful merging occurs and a decline as the threshold becomes too strict to merge variants.

| Threshold $\tau$ | Canonical vocabulary | P@1 |
|:---|:---|:---|
| 0.80 | 418 | 0.450 |
| 0.85 | 459 | 0.450 |
| 0.90 | 526 | 0.350 |
| 0.92 | 531 | 0.350 |

: Effect of the merge threshold on approximate phrase matching.

At $\tau \geq 0.90$ the vocabulary stays near its exact size of 535, few variants merge, and performance falls back to the level of exact matching. At $\tau$ of 0.80 and 0.85 real variants merge, the vocabulary drops to between 418 and 459 terms, and the full gain appears. The improvement thus tracks the amount of merging rather than one tuned value, which is the signature of a real effect. We report $\tau = 0.85$ as the operating point, the least aggressive threshold that achieves the full gain.

## 7.5 Interpretation

Given the small query set, we emphasise direction and mechanism over exact values. The mechanism unifies this positive result with the paper's negative ones. Svara-level retrieval is limited not because structure is unhelpful but because the svara is too coarse and too universal a unit: every raga uses every pitch position, so presence carries nothing and lexical weighting has nothing to weight. Phrases are the unit at which raga identity becomes discriminative, and at which classical IR machinery, term weighting, vector space retrieval and approximate matching, regains its power. The limiting factor for phrase representations is instead transcription consistency, which approximate matching partially resolves.

This positions the study's overall finding not as a statement that learned representations fail but as a claim about representational granularity: at this data scale, the unit of representation matters more than the sophistication of the model. Count-based methods win among svara representations, and the learned svara-sequence models sit below them, having learned the shared grammar of Carnatic melody rather than the phrase-level structure that distinguishes ragas. On the subset where expert phrases exist, a count-based model over phrases does better than a count-based model over svaras.

## 7.6 Limitations of the sub-study

The phrase set is small (60 queries, 22 ragas), and its confidence intervals are correspondingly wide. We make no cell-precise claims. Phrase annotations are themselves expert-produced and finite in coverage, available for 100 of the 249 tracks, so this representation is not available label-free at query time in the way the svara pipeline is. It demonstrates what phrase-level structure contains, not a deployable label-free system. Extending expert or automatically segmented phrase vocabularies to the full collection, and testing whether learned phrase discovery can approximate the expert annotations, are natural next steps.

# 8 Discussion

## 8.1 Why counting is hard to beat at this scale

The pitch-class histogram has twelve parameters and the transition matrix has 144, and both are estimated directly from the data with no fitting procedure. A concert recording provides thousands of note events, so those parameters are estimated from a great deal of evidence and are correspondingly stable. The learned representations, by contrast, must estimate tens of thousands of parameters from 176 recordings, and they must do so through an objective that is only indirectly related to the retrieval task. The comparison is therefore not so much simple against sophisticated as well-estimated against poorly estimated.

This reading is supported by the direction of the errors. The learned models do not fail randomly. They place a disproportionate share of their wrong retrievals in the correct region of the raga space (Section 6.6), which is what a representation with the right general shape but insufficient resolution would do.

## 8.2 Objective misalignment

The generative language models were trained to predict the next svara. Doing that well requires learning the regularities of Carnatic melody in general: that motion is predominantly stepwise, that sa and pa are frequent and structurally important, that certain intervals are rare, that phrases resolve in characteristic ways. All of this is shared across ragas. Almost none of it distinguishes one raga from another. The objective rewards exactly the knowledge that the retrieval task cannot use. The same gap is documented in text retrieval, where language models pretrained with a token-prediction objective are not structurally ready to serve as dense retrievers without further training aimed at the pooled representation [@gao2021condenser].

Three independent observations support this diagnosis. First, the Transformer is weak in the same way as the recurrent model, and more so, which locates the problem in the objective rather than the architecture. Second, the language-modelling metric and retrieval quality move in opposite directions under denoising (Section 6.3): the corpus on which the models predict best is the one on which they retrieve worst, so doing better at the generative objective is not the same as building a better retrieval representation. Third, under graded relevance the errors of every representation fall on allied ragas three to four times more often than chance (Section 6.6). The models know the neighbourhood and not the address.

Contrastive fine-tuning was the targeted response to this diagnosis. On the evidence of five seeds it does not improve on the language model it starts from (0.366 against 0.369), and the likely reason is structural rather than a matter of tuning. The objective treats every other recording in a batch as a negative, but on a collection where the evaluable ragas are precisely the multi-recording ones, some of those negatives are same-raga recordings. The objective actively separates pairs that the evaluation requires together [@chuang2020debiased]. Any label-free contrastive scheme on a class-dense collection faces this, and it bounds what the method can achieve here.

## 8.3 The pooling explanation, tested and eliminated

An obvious objection to all of the above is that the problem lies not in the objective but in mean pooling, which compresses a long sequence into a single vector and may discard exactly the fine structure that raga identity depends on. Late interaction exists to remove that bottleneck. It retains one embedding per token and scores by taking, for each query token, its best match anywhere in the document.

Trained properly under our protocol, with validation loss falling from 17.1 to 1.9 on held-out artists, late interaction retrieved at 0.301, below every run of the mean-pooled language model it was built from (0.358 to 0.398 across seven runs). The training objective was achieved and the retrieval quality went down. Pooling was therefore not the binding constraint. Late interaction can only expose information that the encoder represented in the first place, and the encoder had learned shared grammar.

There is also a reason to expect late interaction to be poorly suited to this domain, which connects it to the BM25 result. MaxSim asks each query token to find its single best match in the document. Over a twelve-symbol vocabulary in which every symbol occurs in every recording, almost every query token can find a close match in almost every document, so the scores compress and discrimination suffers. This is the same property that made document frequency uninformative. Two quite different mechanisms from text retrieval, rare-term weighting and best-match interaction, are undermined here by the same underlying fact: the alphabet is small and every symbol is universal. We offer this as a hypothesis consistent with the result rather than as a tested mechanism.

## 8.4 The unit of representation

Taken alone, the results above read as a negative finding about learned representations. The phrase sub-study of Section 7 changes that reading. On the phrase-annotated subset, a term-weighted vector space model over expert-annotated phrases retrieves at 0.333 against 0.150 for svara counting on the same recordings, and at 0.450 once near-duplicate phrase transcriptions are merged by approximate string matching.

The mechanism is the mirror image of the BM25 failure. Phrases are rare and raga-specific: within a raga, annotated phrase sets overlap at a Jaccard of 0.018, while across ragas the overlap is essentially zero, a ratio of about 57. Inverse document frequency, which carried no information over svaras, carries a great deal over phrases. The classical retrieval machinery was never the problem. It was being applied to the wrong unit.

We therefore state the finding as a claim about representational granularity rather than about model families. At this data scale, the unit of representation matters more than the sophistication of the model. Svaras are too coarse and too universal: every raga uses every pitch position, presence is uninformative, and lexical weighting has nothing to weight. Phrases are the level at which raga identity becomes discriminative. The learned sequence models acquired the grammar that svara sequences share without recovering the phrase-level structure that separates ragas.

The limiting factor for phrase representations is not the idea but the annotation. Phrases are expert-transcribed, cover 100 of the 249 tracks, and are not available for an unlabelled query recording, so the phrase result demonstrates what phrase-level structure contains rather than a deployable label-free system. It also suggests that transcription consistency costs real performance: merging variant transcriptions of the same gesture recovered 0.12 in P@1, which implies that automatic phrase segmentation would need to be more consistent than human transcription is, not merely as good.

## 8.5 Perplexity is not a proxy for musical competence

Denoising the symbolic layer roughly doubled retrieval for the language models and raised their perplexity from about 3.9 to a little above 7. The noise that denoising removed, oscillation between adjacent quantisation bins, was the most predictable content in the corpus. A model selected on perplexity would prefer the noisier data and the worse representation.

We raise this because perplexity is a convenient measure in symbolic music modelling, and because the failure mode is not obvious. It is not that perplexity is a weak proxy for retrieval quality. On this data it moved in the opposite direction.

## 8.6 Relation to prior findings

The pattern we report is not unique to this domain. In raga identification for Hindustani music, a convolutional-recurrent feature extractor trained from scratch outperformed large pretrained music transformers on the same task [@singh2026discover]. In text retrieval, the difficulty of beating simple, well-tuned baselines and the tendency of single results to overstate neural models are long-documented [@armstrong2009improvements; @lin2018neuralhype; @yang2019neuralhype]. Our contribution to that line is a controlled comparison in which the simpler representation wins, together with a mechanism for each gap rather than an observation that the gap exists.

## 8.7 Limitations

The collection is small and single-source, and the results should be read as scale-specific. We make no claim that count-based representations would remain superior on a corpus an order of magnitude larger. The mechanisms we identify, estimation noise, objective misalignment and negative contamination, are all expected to weaken as data grows.

Multi-seed evaluation covers the LSTM language model and contrastive fine-tuning only. The Transformer and late-interaction figures are single runs, and Section 6.2 shows that single runs at this scale move by several points. The gaps that carry the paper's conclusions (0.423 against 0.276 and 0.301) are large relative to that variation, but the exact values are not to be leaned on. The contrastive configuration also differed between the development run and the multi-seed harness, as Section 6.2 sets out.

The v1 figures of Section 6.3 come from development runs made before the dataset was frozen. The learned-model values are single runs, and the count-based aggregates are reconstructed from stratified results rather than measured directly.

The raga theory table simplifies each raga to one arohana and one avarohana. Many ragas admit several accepted forms, bhashanga usage or vakra movement, and 62 of the 95 entries are flagged as medium confidence and 7 as low for that reason. The scale-overlap statistic, and with it the allied stratum and the graded relevance judgements, inherits this simplification.

Enharmonic folding discards a real distinction, as Section 3.2 sets out, and it does so in exactly the place where the paper argues the signal lives. The gamaka descriptors we extract and release are the natural route to recovering it, and we have not used them.

The phrase sub-study rests on 60 queries and 22 ragas, and its confidence intervals are correspondingly wide. We report directions and mechanisms there rather than precise values.

Finally, the collection is Carnatic. Whether these findings transfer to Hindustani music, where phrase structure and ornamentation differ, is an open question that the parallel Saraga Hindustani collection would allow to be tested directly.

## 8.8 What would change the result

Three things, in order of expected effect. More data, since every mechanism we identify is a small-sample mechanism. Phrase-level units learned rather than annotated, since the sub-study shows the unit matters and expert annotation does not scale. And gamaka-aware features, since the distinctions that frame-level pitch cannot make are precisely the ones that ornamentation carries.

# 9 Conclusion

We set out to test whether melodic representations learned without supervision would outperform simple counting for label-free raga retrieval, and in particular whether they would resolve allied ragas, where note content alone cannot. On 176 recordings spanning 95 ragas, they did not. A twelve-dimensional pitch-class histogram and a first-order svara-transition matrix outperformed a recurrent language model, a Transformer language model, contrastive fine-tuning and late interaction, both overall and within the allied stratum. Lexical scoring failed entirely, because a twelve-symbol alphabet in which every symbol occurs in every recording carries no document-frequency information.

The value of the result lies in the accompanying mechanisms. Generative pretraining on svara sequences teaches a model what ragas have in common rather than what distinguishes them, a diagnosis supported by the shared weakness of two architectures, by perplexity and retrieval quality moving in opposite directions under denoising, and by graded relevance judgements showing that every representation errs towards the correct scale neighbourhood far more often than chance. Removing the pooling bottleneck through late interaction made retrieval worse rather than better, which eliminates the most plausible competing explanation. And apparent parity between contrastive fine-tuning and counting in one run was not reproduced in five, which is a cautionary result about evaluation practice at this data scale rather than about any model in particular.

The phrase sub-study shows what was missing. Over expert-annotated phrases rather than individual svaras, term-weighted retrieval outperforms counting, and improves further once variant transcriptions are reconciled by approximate matching. The retrieval machinery that failed over svaras succeeds over phrases because phrases are rare and raga-specific while svaras are universal. We therefore read the study as a claim about representational granularity: at this scale, the unit of representation matters more than the sophistication of the model.

Three directions follow. The first is scale, since every mechanism we identify is a small-sample mechanism and should weaken as data grows. The second is learned phrase discovery, since the sub-study shows the unit matters while expert annotation covers less than half the collection and cannot be produced for an unlabelled query. The third is ornamentation, since the enharmonic distinctions that frame-level pitch cannot make are exactly the ones that gamaka carries, and the gamaka descriptors we extract and release remain unused.

We release the frozen symbolic dataset, the raga theory tables, the evaluation protocol and all code, in the hope that the next study of this question begins from a measured baseline rather than an assumption.

# Data and Code Availability {-}

The frozen symbolic dataset (tonic-normalised svara sequences with per-note gamaka descriptors), the raga theory tables, all extraction, training and evaluation scripts, the trained checkpoints and embeddings, and the result files behind every table are released in the SwaraLM repository, which is archived on Zenodo alongside this preprint. Code is released under the MIT licence. The symbolic dataset is derived from Saraga Carnatic and is released under the same CC BY-NC-SA 4.0 licence. Audio is not redistributed and is available from the maintainers of Saraga [@bozkurt2018saraga].

# Use of AI Assistance {-}

The author chose the problem, ran every experiment, supplied the raga theory table and the musical domain knowledge, identified and corrected errors in the extraction and in the musical claims, and initiated the multi-seed evaluation, the late-interaction comparison and the phrase sub-study. An AI assistant (Claude, Anthropic) was used throughout the project under the author's direction: it implemented the analysis code, contributed to the experimental design, and drafted the text of this paper, which the author reviewed and corrected. The author takes responsibility for the content.

# Acknowledgements {-}

This study depends entirely on the Saraga datasets and the wider CompMusic corpora, and on the mirdata loaders. The author thanks the musicians whose performances make up the collection and the researchers who annotated and released it.

# References {-}

::: {#refs}
:::
