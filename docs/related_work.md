# Related Work — Automatic News Topic Classification

## Overview
This document reviews 10 key research papers and resources directly relevant to this project.

---

## 1. Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features
**Authors:** Thorsten Joachims  
**Year:** 1998  
**Venue:** European Conference on Machine Learning (ECML)  
**Problem:** Efficient text classification using SVMs  
**Method:** Linear SVM with TF-IDF features  
**Dataset:** Reuters-21578  
**Result:** SVMs outperformed previous approaches on text categorization  
**Relevance:** Establishes SVM as a strong baseline for text classification; directly compared with Logistic Regression in this project.

---

## 2. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification
**Authors:** Xiang Zhang, Junbo Zhao, Yann LeCun  
**Year:** 2015  
**Venue:** Advances in Neural Information Processing Systems (NeurIPS)  
**Problem:** Text classification at character level  
**Method:** Deep character-level CNNs  
**Dataset:** AG News (among others) — 4 classes, 120K train samples  
**Result:** Competitive accuracy on AG News (~91.4%)  
**Relevance:** This paper introduced the AG News dataset used in our project and established benchmark results.

---

## 3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
**Year:** 2019  
**Venue:** NAACL-HLT 2019  
**Problem:** General-purpose NLP representation learning  
**Method:** Masked Language Modeling + Next Sentence Prediction pre-training  
**Dataset:** BooksCorpus, Wikipedia  
**Result:** State-of-the-art on 11 NLP tasks  
**Relevance:** Foundation of the DistilBERT model used in this project.

---

## 4. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
**Authors:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf  
**Year:** 2019  
**Venue:** NeurIPS EMC² Workshop  
**Problem:** Making BERT practical for resource-constrained environments  
**Method:** Knowledge distillation from BERT  
**Dataset:** GLUE benchmark  
**Result:** 97% of BERT performance, 40% smaller, 60% faster  
**Relevance:** Direct motivation for choosing DistilBERT as the transformer model in this project.

---

## 5. Joachims, T. (1996). A Probabilistic Analysis of the Rocchio Algorithm with TFIDF for Text Categorization
**Authors:** Thorsten Joachims  
**Year:** 1996  
**Venue:** ICML 1997  
**Problem:** Theoretical analysis of TF-IDF for text classification  
**Method:** Probabilistic analysis  
**Dataset:** Reuters  
**Result:** Proved TF-IDF effectively separates class distributions  
**Relevance:** Theoretical justification for TF-IDF feature extraction used in the baseline model.

---

## 6. Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., & Gao, J. (2021). Deep Learning–Based Text Classification: A Comprehensive Review
**Authors:** Shervin Minaee et al.  
**Year:** 2021  
**Venue:** ACM Computing Surveys  
**Problem:** Survey of deep learning methods for text classification  
**Method:** Comprehensive literature review  
**Dataset:** Multiple (AG News, Reuters, SST, etc.)  
**Result:** Transformers dominate modern text classification  
**Relevance:** Provides context for choosing transformers over CNNs/RNNs for this project.

---

## 7. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
**Authors:** Yoon Kim  
**Year:** 2014  
**Venue:** EMNLP 2014  
**Problem:** Sentence-level text classification  
**Method:** CNNs on word embeddings  
**Dataset:** MR, SST, CR, TREC, MPQA  
**Result:** Strong results with simple CNN architecture  
**Relevance:** Represents the CNN approach comparison point for news classification.

---

## 8. Wang, S., & Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification
**Authors:** Sida Wang, Christopher Manning  
**Year:** 2012  
**Venue:** ACL 2012  
**Problem:** Efficient NLP classification  
**Method:** Naive Bayes and SVM with bigrams  
**Dataset:** Multiple  
**Result:** Simple n-gram models remain competitive  
**Relevance:** Motivates n-gram features (1,2) in TF-IDF configuration.

---

## 9. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-Tuning for Text Classification
**Authors:** Jeremy Howard, Sebastian Ruder  
**Year:** 2018  
**Venue:** ACL 2018  
**Problem:** Transfer learning for NLP  
**Method:** ULMFiT — pre-train LM, fine-tune for classification  
**Dataset:** Multiple classification datasets  
**Result:** State-of-the-art with less data  
**Relevance:** Foundational work motivating fine-tuning pre-trained models (like DistilBERT) for news classification.

---

## 10. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python
**Authors:** Fabian Pedregosa et al.  
**Year:** 2011  
**Venue:** Journal of Machine Learning Research (JMLR)  
**Problem:** Unified Python ML library  
**Method:** Scikit-learn library  
**Dataset:** N/A (framework paper)  
**Result:** Widely adopted, reproducible ML in Python  
**Relevance:** The scikit-learn library is used for TF-IDF vectorization, Logistic Regression, and evaluation metrics in this project.

---

## Bibliography

1. Joachims, T. (1998). Text Categorization with Support Vector Machines. ECML 1998.
2. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS 2015.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.
4. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. NeurIPS Workshop 2019.
5. Joachims, T. (1997). A Probabilistic Analysis of the Rocchio Algorithm with TFIDF. ICML 1997.
6. Minaee, S. et al. (2021). Deep Learning–Based Text Classification: A Comprehensive Review. ACM Comput. Surv.
7. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP 2014.
8. Wang, S., & Manning, C. D. (2012). Baselines and Bigrams. ACL 2012.
9. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-Tuning. ACL 2018.
10. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.
