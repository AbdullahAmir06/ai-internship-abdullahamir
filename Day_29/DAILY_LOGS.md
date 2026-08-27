# Daily Logs — PKCERT AI & Software Development Internship

Compiled from this repository's own commit history and each day's README, which together
constitute the contemporaneous record of what was built and delivered each day. Every day
1–28 has a corresponding folder in this repository with its own code, report, and (from Day
23 onward) an executed Jupyter notebook — see `Day_XX/README.md` for that day's full detail.

| Day | Topic | Key deliverable |
|---|---|---|
| 1–3 | Environment setup, personal website, early task briefs | Local dev environment, deployed static website |
| 4 | Titanic dataset — exploratory data analysis | Pandas/NumPy EDA report |
| 5 | NYC Airbnb: preprocessing, visualization, feature engineering & SQL | Cleaned dataset + SQL feature pipeline |
| 6 | EDA: Red Wine Quality | Jupyter notebook, full EDA workflow |
| 7 | ML foundations: train/test split, linear regression | First trained regression model |
| 8 | Classification models: logistic regression, decision trees, random forests | Multi-model classification comparison |
| 9 | SVM & k-NN: classifying raisin varieties | Kernel-method classifiers |
| 10 | Cross-validation & hyperparameter tuning: pulsar detection | Systematic hyperparameter search |
| 11 | Comparing classification models: student dropout prediction | Logistic Regression vs. Random Forest vs. SVM |
| 12 | Clustering & dimensionality reduction: frog species from audio (MFCCs) | Unsupervised learning pipeline |
| 13 | Advanced model evaluation & imbalanced data: credit card fraud | Precision/recall-focused evaluation under class imbalance |
| 14 | Ensemble methods: bagging & boosting | Bank telemarketing response prediction |
| 15 | Model persistence & mini-project: heart disease prediction | Serialized, reloadable model |
| 16 | End-to-end ML pipeline: Ames Housing price prediction | Full pipeline capstone (pre-deep-learning) |
| 17 | Neural network fundamentals: perceptron, activations, manual backprop | Hand-derived backpropagation |
| 18 | Intro to PyTorch: tensors, autograd, `nn.Module` | First PyTorch model |
| 19 | Training loops: loss functions, SGD/Adam from scratch, batch/epoch management | Custom training loop engineering |
| 20 | Feedforward neural network on Fashion-MNIST | Image classification with a dense network |
| 21 | Regularization techniques in deep learning | Dropout/weight-decay/early-stopping comparison |
| 22 | Feedforward NNs on MNIST, regularization, GPU-accelerated training | GPU training pipeline |
| 23 | Convolutional neural networks from scratch | NumPy-only CNN, gradient-checked, matched a PyTorch baseline within 0.25 points |
| 24 | CNN & transfer learning: CIFAR-10, custom CNN vs. pretrained architectures | ResNet/VGG/MobileNet transfer learning, 84.5% best accuracy |
| 25 | Sequence modeling: RNN/LSTM fundamentals & text classification | From-scratch LSTM cell (gradient-checked vs. `nn.LSTMCell`), AG News classifier |
| 26 | Introduction to Transformers & attention (conceptual) | Attention/Transformer derivations, applied DistilBERT vs. Task 25's LSTM |
| 27 | NLP basics: tokenization, Word2Vec/GloVe, Hugging Face embeddings | From-scratch BPE tokenizer, static-vs-contextual embeddings comparison |
| 28 | Production LLM serving, async microservices & full-stack deployment | FastAPI microservice, Docker, real deployment attempt (Render OOM finding, documented) |
| 29 | **Capstone**: Phishing Email Inspection Desk | This project — see `FINAL_REPORT.md` |

## Notes on how these logs were compiled

Each row above is drawn directly from that day's actual commit message and README — not
reconstructed from memory — so this table is traceable back to `git log` and the
corresponding `Day_XX/` folder for verification. Days 1–22 (classical ML and early deep
learning fundamentals) predate this session's own involvement; Days 23–29 were completed
within it, and their entries reflect the actual measured results reported in each day's own
`README.md`/`Report.pdf` at the time.
