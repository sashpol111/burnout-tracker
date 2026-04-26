# Attribution

## AI Development Tools

Claude (Anthropic) was used as a collaborative development tool in specific parts of this project. Below is an account of what was generated, what was modified, and what required independent work.

### Where Claude Was Used

**Deployment to Hugging Face Spaces**
Claude was the primary resource for deploying the app. This required significant debugging: resolving binary file rejections from git history using git filter-branch, fixing YAML metadata errors in README.md, restructuring the app to retrain the model on startup instead of loading saved .pkl files, and troubleshooting Duke cluster port forwarding issues. The deployment process was iterative and required substantial human judgment to resolve each blocker.

**Error Analysis (src/error_analysis.py)**
Claude helped scaffold the confusion matrix plotting and false negative/positive profile structure. The analysis design, interpretation of failure cases, and discussion of what the errors mean for burnout prediction were done independently.

**Visualizations (src/visualize.py)**
Claude helped with the initial matplotlib plotting code for feature importance and model comparison charts. Chart design choices and the decision of what to visualize were made independently.

**Ablation Study (src/ablation.py)**
Claude helped with the script structure. The experimental design — which feature sets to test and why — was designed independently based on the correlation analysis and domain knowledge about burnout.

## Data Sources
- Kaggle Wellness and Lifestyle Dataset: https://www.kaggle.com/datasets/ydalat/lifestyle-and-wellbeing-data
- HuggingFace: solomonk/reddit_mental_health_posts
- Maslach, C., Jackson, S. E., & Leiter, M. P. (1996). Maslach Burnout Inventory Manual.

## Libraries and Frameworks
XGBoost, PyTorch, scikit-learn, Streamlit, Groq API, HuggingFace Datasets, Transformers, pandas, numpy