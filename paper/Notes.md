Notes:

a few notes. 1. confirm thesee values are correct "Class distribution comprised inflam￾matory (I: 23.25%), proliferative (P: 63.21%), and remodeling
(R: 13.09%) phases,", "Metadata processing uses 72 clinical features", also later in the paper it is mentioned "Several limitations warrant acknowledgment. Class imbalance (I: 23%, P: 63%, R: 13%)," which is a different decimal place. is it fine? if yes, keep it as is.
2. remove this part "and
retention in the [0, 255] range (the DenseNet121 backbone
includes built-in rescaling)." too much coding details.
3. Random Forest (RF) should be abberivated at the first instance which is in "B. Dataset and Preprocessing" section, and from there on use RF such as in section "1) Modality-Specific Processing:".
4. this "a Dense(3, softmax) output layer" is too much coding info or language. maybe just say it in less CS lnaguage? could we say a neuronal network layer with 3 possible outcomes or something better? we could possibly still use Dense layer but just remove "(3, max)".
5. "early stopping with patience 20" too much code specefic info in the journal article intended for biomedical engineer domain.
6. remove " If Stage 2 does not improve validation kappa, Stage 1
weights are restored." to much code details.
7. in "simple
averaging, optimal weighted averaging, temperature scaling,
logistic regression stacking, multilayer perceptron (MLP)
stacking, gradient boosting stacking, rank-weighted averaging,
and multi-head attention" if the claim is each of these are different strategies, then there should be citation for each method.
8. remove this detail "(144 images per phase, 432 total)."
9. instead of " at a configurable probability
with 1-5% mix ratio per batch" just use the mix ratio that was used. or you can remove this detail alltogheter in case mix ratio is not a common knoweledge and need to be explained.
10. all these names and methods need to be cited if not already done earlier, " Using Bayesian
optimization with Gaussian Process surrogate and Expected
Improvement acquisition (100 trials), the search co-optimized
23 hyperparameters spanning image backbones (DenseNet121,
EfficientNetB0/B2, ResNet50V2, MobileNetV3Large),"
11. in my previouse paper i explored the independent configuration related to this portion of the paper " independently tuned configuration (EfficientNetB0
at 256×256", so my previouse paper should be cited wherever this claim is mentioned or related to it. my previouse paper is "" in case if not already in the biblography. "R. Basiri, M. R. Popovic and S. S. Khan, "Domain-Specific Deep Learning Feature Extractor for Diabetic Foot Ulcer Detection," 2022 IEEE International Conference on Data Mining Workshops (ICDMW), Orlando, FL, USA, 2022, pp. 1-5, doi: 10.1109/ICDMW58026.2022.00041."  
12. This "This joint approach discovered that DenseNet121 at
128×128 resolution with feature concatenation fusion outper￾forms the independently tuned configuration (EfficientNetB0
at 256×256), improving five-fold mean kappa from 0.30 to
0.37. The improvement arises because modality interactions
during fusion create dependencies that per-modality optimization cannot capture." is a result, and it should not be written in the Methods section.
13. so did we decide to do one decimal place for values for whole numbers or 2, cuz here " with accuracy of 70.55%, followed
by RGB (kappa 0.51 ± 0.07, accuracy 60.96%), thermal
(kappa 0.50 ± 0.10, accuracy 58.13%), and depth (kappa
0.19 ± 0.05, accuracy 44.21%)." "accuracy of 71.24%." "e highest accuracy (78.50%)" " maintaining accuracy of 80.56%." "and
accuracy of 75.38%" "with ensemble accuracy of 78.86%, " "alone still achieves 70.55% accuracy." "of 75.38% in three-class healing
phase classification" "parameter-free ensemble framework achieving
80.56% accuracy" i see two. else where i think i saw 1 decimal place.
14. regarding this point "The modality contribution analysis revealed a clear imple￾mentation hierarchy: metadata provides the strongest stan￾dalone signal (kappa 0.45)," and other point about metadata similar to this. in my previouse papers, i showed the power of metadata in trajectory of wound healing prediction which is inline with this finding and should be cited. here is my previouse paper "Basiri, R., Saleh, A., Khan, S.S. et al. Temporal machine learning framework for diabetic foot ulcer healing trajectory prediction. BioMed Eng OnLine 25, 41 (2026). https://doi.org/10.1186/s12938-026-01529-2", if not already in the biblography doc. 
15. This "Second, reducing input resolution from 256×256 to 128×128
pixels improved performance, because the average wound
bounding box is only 70×70 pixels; at 256×256, the 3-14×
upsampling introduces interpolation artifacts that obscure
genuine wound texture." is very dataset dependable so not a strong point to mention in the discussion, can be removed.
16. Something i want you to note and reflect in the paper where appropirate is that these values should be mainly studied and looked at relative to each other, so we get a clear picture of how modalities combine and perform on a uniform dataset with same number of samples for each modality. While the absulote values provide some information about the performance, those can easily be improved by additional dataset samples which is not the point of this study. the absulote values depend to the data size so when comparing to other studies we should be carful. and the point of this paper is not to come up with the model that gives the highest absolute performance values but to shed light on multimodality in a fair and uniform setup.
17. this are too much details and not so relavent and can be removed "The pre-caching approach (generating
synthetic images to disk before training) provides a prac￾tical solution for dual-framework environments where Py￾Torch-based generation and TensorFlow-based training cannot
share GPU memory. The decision to limit augmentation to
RGB imagery reflects important considerations regarding the
measurement-based nature of thermal and depth modalities."
18. in my previouse papers, i have used generative ai for application in DFU. while not SDXL, i used SD model which should be cited in the "D. Generative Augmentation and Technical Innovation" where applicable very briefly. here is my previouse paper "Basiri, R., Ghaffar, A., Ghiasi, D., Mekonnen, M.T., Popovic, M.R., Khan, S.S. (2025). Enhancing Diabetic Foot Ulcer Assessment Through Fine-Tuned Vision-Language Models. In: Khan, S.S., Romeo, L., Abedi, A. (eds) ArtifiAI for Aging Rehabilitation and Intelligent Assisted Living. IJCAI 2025. Communications in Computer and Information Science, vol 2620. Springer, Singapore. https://doi.org/10.1007/978-981-95-0568-5_9" and "Basiri, R., Manji, K., Francois, H., Poonja, A., Popovic, M.R., Khan, S.S. (2025). Synthesizing Diabetic Foot Ulcer Images with Diffusion Model. In: Meo, R., Silvestri, F. (eds) Machine Learning and Principles and Practice of Knowledge Discovery in Databases. ECML PKDD 2023. Communications in Computer and Information Science, vol 2136. Springer, Cham. https://doi.org/10.1007/978-3-031-74640-6_28". if not already included.
19. also i have used LLMs in the application of DFU. while not directly relavent but it is still generative and should be cited where most appropirate. here is my previouse paper "Basiri, R., Abedi, A., Nguyen, C., Popovic, M.R., Khan, S.S. (2025). UlcerGPT: A Multimodal Approach Leveraging Large Language and Vision Models for Diabetic Foot Ulcer Image Transcription. In: Palaiahnakote, S., Schuckers, S., Ogier, JM., Bhattacharya, P., Pal, U., Bhattacharya, S. (eds) Pattern Recognition. ICPR 2024 International Workshops and Challenges. ICPR 2024. Lecture Notes in Computer Science, vol 15618. Springer, Cham. https://doi.org/10.1007/978-3-031-88220-3_16"
20. remove this " (58 samples after screening)"
21. regarding " (approximately
$3,000-$5,000 per unit)", you cant just throw a number without citations. it is best to just remove the $$ values.
22. " The generative augmentation cache size (144
images per phase) constrains the maximum useful injection
probability; larger caches may extend the useful augmentation
range." this is irrelevent and very minor. we only did it so it can run on my system. remove it.
23. in " constrains the maximum useful injection
probability; larger caches may extend the useful augmentation
range." remove "caches".
24. the "Experimental validation on 443 wound assessments from" or in general the 443 value undermind the datasize as people might think that is the total datasize. but the datasize was larger when you include the multiple images. lets re-state the "3,108" value instead of 443
25. remove "We are actively working towards
public dataset availability, navigating through challenges
and dedicating substantial time and resources to make this
possible. We plan to facilitate the release of the dataset
to selected research institutes upon request after ethics
amendment approvals."