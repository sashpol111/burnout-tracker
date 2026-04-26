## AI Development Tools

Claude (Anthropic) was used as a collaborative development tool in specific parts of this project. Below is an account of what was generated, what was modified, and what required independent work.

### Where Claude Was Used

**Deployment to Hugging Face Spaces**
Claude was the primary resource for deploying the app. This included debugging git history issues with large binary files, fixing YAML metadata errors in README.md, restructuring the app to retrain the model on startup instead of loading saved .pkl files, and troubleshooting Duke cluster port forwarding. The deployment process was iterative and required manual debugging at each step.

**LLM Advisor (src/llm_advisor.py)**
Claude assisted with structuring the LLM integration, including Groq API calls, prompt formatting, and multi-turn conversation handling. Context management and integration into the app flow were refined through additional debugging.

**Preprocessing Experiment (src/preprocessing_experiment.py)**
Claude helped with initial code structure for running preprocessing and regularization comparisons. Several issues in earlier versions of the pipeline led to revisions in how experiments were defined and evaluated. The final setup reflects changes made during iteration with my project partner after identifying these issues.

**Streamlit App (app.py)**
Claude assisted with early versions of the Streamlit interface. As the project evolved, the app was reworked to match changes in the modeling pipeline and LLM integration. This included restructuring inputs, outputs, and how results are passed into the advisor.

**Error Analysis (src/error_analysis.py)**
Claude helped scaffold the confusion matrix plotting and false negative and false positive grouping structure. The analysis design and interpretation of results were completed independently.

**Visualizations (src/visualize.py)**
Claude helped with initial matplotlib code for feature importance and model comparison plots. Visualization choices and selection of metrics were determined independently.

**Ablation Study (src/ablation.py)**
Claude helped with the script structure. The experimental setup, including feature group selection, was designed independently based on prior analysis.