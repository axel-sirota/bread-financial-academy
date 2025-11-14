# Bread Financial - AI for Data Scientists Academy

Welcome to the **Bread Financial AI Academy** repository! This comprehensive 24-week program provides hands-on training in deep learning, MLOps, GenAI, and production ML systems for data scientists.

## 📚 Program Overview

- **Duration**: 24 weeks (delivered over 12 months)
- **Format**: 2-hour weekly hands-on lab sessions
- **Students**: 60 participants across 3 cohorts
- **Philosophy**: Flipped classroom - students watch theory videos before class, sessions focus on practical application

## 🗂️ Repository Structure

```
bread-financial-academy/
├── exercises/              # Student notebooks (distributed before class)
│   ├── week_01_pytorch_basics/
│   ├── week_02_cnns_rnns/
│   └── ... (24 weeks total)
├── solutions/              # Complete solutions (shared after class)
│   ├── week_01_pytorch_basics/
│   ├── week_02_cnns_rnns/
│   └── ... (24 weeks total)
├── datasets/               # Dataset documentation and references
├── infrastructure/         # Terraform and cloud infrastructure setup
├── initial_docs/           # Course outline and technical specifications
├── CLAUDE.md              # Development guide for creating materials
└── README.md              # This file
```

## 📅 Curriculum Overview

### Deep Learning Foundations (Weeks 1-2)
- **Week 1**: PyTorch Basics - Tensors, autograd, neural networks
- **Week 2**: CNNs & RNNs - Transfer learning, sequence modeling

### ML on Databricks (Weeks 3-4)
- **Week 3**: Spark & Regression - DataFrames, distributed ML pipelines
- **Week 4**: Spark & Classification - Model selection, hyperparameter tuning

### ML on AWS (Weeks 5-7)
- **Week 5**: SageMaker Basics - Training jobs, endpoints, built-in algorithms
- **Week 6**: Neural Networks & Tuning - Custom scripts, hyperparameter optimization
- **Week 7**: MLflow & Observability - Experiment tracking, model monitoring

### Git & Development Workflows (Weeks 8-10)
- **Week 8**: Git Workflows & Collaboration - Gitflow, merge conflicts, team workflows
- **Week 9**: SDLC & TDD - Test-driven development, Copilot best practices
- **Week 10**: Code Reviews & Hotfixes - Pull requests, emergency response

### Large Language Models (Weeks 11-14)
- **Week 11**: LLM Fundamentals - HuggingFace, tokenization, prompting strategies
- **Week 12**: GenAI for Data Science - Synthetic data, evaluation, LLM-assisted analysis
- **Week 13**: Amazon Bedrock - Model selection, pricing, Knowledge Bases
- **Week 14**: Training AI Models - Fine-tuning with LoRA/QLoRA, distillation

### Agentic AI (Weeks 15-16)
- **Week 15**: Developing Agentic AI - ReAct pattern, LangChain, memory systems
- **Week 16**: Integrating Agentic AI - LangGraph workflows, multi-agent systems

### Retrieval-Augmented Generation (Weeks 17-18)
- **Week 17**: RAG Fundamentals - Embeddings, vector databases, basic RAG pipelines
- **Week 18**: RAG Advanced - Chunking strategies, evaluation, reranking

### MLOps (Weeks 19-20)
- **Week 19**: Versioning & Experiments - DVC, MLflow, model registry
- **Week 20**: CI/CD & Monitoring - Pipelines, drift detection, LLM observability

### Workflow Orchestration (Weeks 21-22)
- **Week 21**: Airflow Fundamentals - DAGs, operators, local development
- **Week 22**: Airflow Production - Remote deployment, executors, best practices

### Ethics & Capstone (Weeks 23-24)
- **Week 23**: AI Governance & Ethics - Bias detection, explainability, regulations
- **Week 24**: Capstone Projects - Student presentations and demonstrations

## 🚀 Getting Started

### For Students

1. **Access Materials**: Clone this repository or access through your learning platform
2. **Week Preparation**:
   - Watch assigned theory videos before the session
   - Review the week's exercise notebook in `exercises/week_XX_topic/`
3. **During Class**:
   - Follow along with instructor demos
   - Complete hands-on labs independently
4. **After Class**:
   - Review solution notebooks in `solutions/week_XX_topic/`
   - Complete optional/extra labs for deeper learning

### For Instructors

1. **Read [CLAUDE.md](CLAUDE.md)** for detailed development guidelines
2. **Prepare Session**:
   - Review exercise and solution notebooks
   - Test all code cells (Restart & Run All)
   - Prepare additional examples as needed
3. **Teaching Workflow**:
   - Share exercise notebook before class
   - Live demo from demo sections
   - Support students during lab time
   - Share solution notebook after class

## 🛠️ Technical Requirements

### Environment by Week

| Weeks | Platform | Notes |
|-------|----------|-------|
| 1-2 | Google Colab / JupyterLab | PyTorch, CNNs, RNNs |
| 3-4 | Azure Databricks | Spark ML |
| 5-7 | AWS SageMaker | Managed ML training |
| 8-10 | Local + GitHub Copilot | Git, SDLC, TDD |
| 11-18 | Google Colab / JupyterLab | LLMs, GenAI, RAG |
| 19-20 | Local + Copilot | MLOps tools |
| 21-22 | AWS Airflow (MWAA) | Orchestration |
| 23 | Google Colab / JupyterLab | AI Ethics |
| 24 | Mixed | Based on capstone |

### Local Setup (for weeks 8-10, 19-20)

**Requirements**:
- Python 3.10 or 3.11
- Git 2.30+
- VS Code with GitHub Copilot extension (or JetBrains with Copilot)
- 8GB RAM minimum (16GB recommended)

**Install Python packages as needed**:
```bash
# Week 8-10: Git/SDLC
pip install pytest pytest-cov black flake8 mypy

# Week 19-20: MLOps
pip install mlflow dvc[s3] langfuse litellm pandas numpy scikit-learn
```

## 📖 Teaching Philosophy

This academy follows a **"show, then do"** approach:

- **Storytelling**: Every topic starts with a real-world problem
- **Demo-driven**: Instructors demonstrate concepts, students apply them
- **Medium difficulty**: Labs are achievable in 15-30 minutes with preparation
- **Public datasets**: All datasets are freely accessible (MNIST, CIFAR, sklearn datasets, HuggingFace)
- **Heavily commented**: Code is a teaching tool with extensive explanations
- **Friendly but professional**: Approachable tone with clear, direct guidance

## 🤝 Contributing

### For Course Developers

1. Read [CLAUDE.md](CLAUDE.md) for detailed guidelines
2. Create feature branch: `git checkout -b week-XX-topic`
3. Develop materials following the notebook template
4. Test thoroughly (Restart & Run All)
5. Create pull request for peer review
6. Iterate based on feedback

### Notebook Checklist

Before submitting new materials:
- [ ] Week title and learning objectives clear
- [ ] Environment setup instructions included
- [ ] Each topic follows: theory → demo → lab structure
- [ ] All code is heavily commented
- [ ] Lab instructions are detailed and step-by-step
- [ ] Real-world context/storytelling present
- [ ] Optional/extra lab included
- [ ] Notebook runs top-to-bottom without errors
- [ ] Solution notebook complete with all code

## 📋 Resources

- **Course Outline**: See [initial_docs/outline.md](initial_docs/outline.md) for detailed weekly breakdown
- **Infrastructure Specs**: See [initial_docs/technical_specs.md](initial_docs/technical_specs.md) for cloud setup
- **Development Guide**: See [CLAUDE.md](CLAUDE.md) for teaching philosophy and standards

## 📞 Support

For questions about:
- **Content/Teaching**: Review CLAUDE.md or consult lead instructor
- **Infrastructure**: Check technical_specs.md or contact DevOps
- **Access Issues**: Contact your program coordinator

## 📄 License

See [LICENSE](LICENSE) file for details.

---

**Let's build practical AI skills through hands-on learning!** 🚀
