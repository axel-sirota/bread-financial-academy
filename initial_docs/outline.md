# **Bread Financial \- AI for Data Scientists Academy**

**Format:** 2-hour weekly sessions (virtual, mentored learning)  
**Philosophy:** Students complete self-paced videos during week, instructor session focuses on hands-on labs and practical application

## **Week 1: Deep Learning \- PyTorch Basics**

**Students arrive with:**

* Python, NumPy, Pandas fundamentals  
* Self-paced theory: neural networks, backpropagation, activation functions

**We teach/practice:**

* PyTorch fundamentals: tensors, autograd, nn.Module  
* Building and training neural networks  
* Complete training loop structure

**Labs:**

1. MNIST digit classification with feedforward network

**Skills developed:**

* Can build and train basic neural networks in PyTorch  
* Understand training loop structure and optimization

**Extra/Optional (async):**

* PyTorch basics lab (tensor operations, autograd)  
* Build feedforward neural network for Iris dataset classification  
* Experimenting with different architectures  
* Advanced autograd mechanics

## **Week 2: Deep Learning \- CNNs & RNNs**

**Students arrive with:**

* Week 1 PyTorch basics, feedforward networks  
* Self-paced videos: CNNs (convolution, pooling, architectures) and RNNs (sequence modeling, hidden states)

**We teach/practice:**

* Transfer learning with pretrained CNNs  
* CNN architecture basics (convolution, pooling)

**Labs:**

1. Transfer learning: Fine-tune pretrained ImageNet model (ResNet/VGG) on custom image dataset

**Skills developed:**

* Can apply transfer learning for computer vision tasks

**Extra/Optional (async):**

* Character-level RNN: Build surname generator  
* Building CNNs from scratch  
* RNN/LSTM sequence modeling fundamentals

## **Week 3: ML on Databricks \- Spark & Regression**

**Students arrive with:**

* PyTorch/Deep Learning basics  
* Self-paced videos: Databricks workspace, Spark DataFrames, Spark ML library basics

**We teach/practice:**

* Working with Spark DataFrames  
* Data preprocessing in distributed Spark environment  
* Building regression models using Spark ML

**Labs:**

1. Build regression pipeline: feature engineering → model training → evaluation (housing price prediction or sales forecasting)

**Skills developed:**

* Can work with Spark DataFrames on Databricks  
* Can build regression pipelines with Spark ML  
* Understand distributed data processing

**Extra/Optional (async):**

* Load and explore data with Spark DataFrames (deep dive)  
* Multiple regression algorithms comparison  
* Advanced feature transformations  
* Spark SQL for data exploration

## **Week 4: ML on Databricks \- Spark & Classification**

**Students arrive with:**

* Week 3 Spark \+ regression experience  
* Self-paced videos: Spark ML classification algorithms, model evaluation metrics

**We teach/practice:**

* Building classification models with Spark ML (Logistic Regression, Random Forest, GBT)  
* Handling imbalanced data  
* Cross-validation and model selection in Spark

**Labs:**

1. Build classification pipeline (customer churn or fraud detection)  
2. Hyperparameter tuning with CrossValidator  
3. Model evaluation and selection

**Skills developed:**

* Can build classification pipelines with Spark ML  
* Can perform model selection and tuning on Databricks  
* Understand distributed ML workflows

**Extra/Optional (async):**

* Gradient Boosted Trees deep dive  
* Advanced cross-validation strategies  
* Handling severe class imbalance (SMOTE, class weights)

## **Week 5: ML on AWS \- SageMaker Basics & Classic ML**

**Students arrive with:**

* Spark/Databricks experience  
* Self-paced videos: AWS fundamentals, S3, SageMaker overview, training instances

**We teach/practice:**

* Setting up SageMaker environment  
* Loading data from S3  
* Training classic ML models on SageMaker training instances  
* Deploying model endpoints for inference

**Labs:**

1. End-to-end ML pipeline: S3 data → SageMaker training job → model deployment → inference  
2. Practice with built-in algorithms (XGBoost on tabular data)

**Skills developed:**

* Can build ML pipelines on SageMaker  
* Understand training instances and endpoints  
* Can deploy models for real-time inference

**Extra/Optional (async):**

* Multiple built-in algorithms exploration  
* Batch transform jobs (alternative to endpoints)  
* SageMaker pricing optimization  
* IAM permissions deep dive

## **Week 6: ML on AWS \- Neural Networks & Hyperparameter Tuning**

**Students arrive with:**

* Week 5 SageMaker basics  
* Self-paced videos: SageMaker hyperparameter tuning, model versioning, training neural networks on SageMaker

**We teach/practice:**

* Training RNNs on SageMaker with custom PyTorch/TensorFlow scripts  
* Running hyperparameter tuning jobs  
* Model registry and versioning best practices

**Labs:**

1. Train RNN model with custom training script on SageMaker  
2. Set up hyperparameter tuning job (optimize RNN architecture/learning rate)  
3. Use model registry for version management

**Skills developed:**

* Can train custom neural networks on SageMaker  
* Can run hyperparameter optimization jobs  
* Can manage model versions effectively

**Extra/Optional (async):**

* Advanced hyperparameter tuning strategies  
* Distributed training across multiple instances  
* SageMaker Processing jobs

## **Week 7: ML on AWS \- MLflow, Observability & Monitoring**

**Students arrive with:**

* Week 6 SageMaker training/tuning experience  
* Self-paced videos: MLflow, SageMaker Model Monitor, CloudWatch

**We teach/practice:**

* Integrating MLflow with SageMaker  
* Setting up model monitoring for data drift and quality  
* CloudWatch metrics and alarms for ML systems

**Labs:**

1. Track experiments with MLflow on SageMaker  
2. Deploy model with monitoring enabled (detect data drift, model performance degradation)  
3. Set up alerts for model issues

**Skills developed:**

* Can use MLflow for experiment tracking on AWS  
* Can implement model monitoring and observability  
* Understand production ML system health metrics

**Extra/Optional (async):**

* Advanced MLflow features (model registry, projects)  
* Custom monitoring metrics  
* Concept drift vs data drift  
* Multi-model monitoring dashboards

## **Week 8: GitHub \+ Copilot \- Git Workflows & Collaboration**

**Students arrive with:**

* All previous ML skills  
* Self-paced videos: Git basics, branching strategies, gitflow, GitHub Copilot features

**We teach/practice:**

* Gitflow branching strategy in ML projects  
* Resolving complex merge conflicts  
* Collaborative workflows for data science teams  
* Using Copilot as an assistant while maintaining understanding of git operations

**Labs:**

1. Work through pre-prepared git scenarios (merge conflicts, feature branches, hotfixes, release branches)  
2. Practice using Copilot to suggest git commands but validate and understand each action  
3. Simulate team collaboration scenarios (pull requests, code reviews)

**Skills developed:**

* Can work with gitflow in ML projects  
* Can resolve complex git situations  
* Can use Copilot effectively while maintaining understanding of git operations  
* Ready for collaborative ML development

**Extra/Optional (async):**

* Alternative branching strategies (trunk-based development)  
* Advanced git commands (rebase, cherry-pick, bisect)  
* GitHub Actions introduction

## **BREAK WEEK after Week 8**

## **Week 9: GitHub \+ Copilot (Part 2\) \- SDLC & Test-Driven Development**

**Students arrive with:**

* Week 8 git workflows and gitflow experience  
* Self-paced videos: Software Development Lifecycle (SDLC), Test-Driven Development (TDD), Copilot best practices, code quality standards

**We teach/practice:**

* SDLC in ML/data science projects  
* Test-Driven Development with Copilot  
* Setting up Copilot rules and guardrails  
* Enforcing good practices (linting, formatting, type hints)  
* Building production-quality applications with Copilot assistance

**Labs:**

1. Build a small ML application using TDD approach (write tests first, use Copilot to help implement)  
2. Configure Copilot workspace rules and coding standards  
3. Practice writing unit tests for ML pipelines with Copilot  
4. Refactor existing code to follow best practices

**Skills developed:**

* Can apply TDD methodology in ML projects  
* Can configure and use Copilot effectively within coding standards  
* Understand SDLC for data science applications  
* Can write production-quality, tested code

**Extra/Optional (async):**

* Advanced testing strategies (integration tests, fixtures)  
* CI/CD introduction  
* Test coverage analysis

## **Week 10: GitHub \+ Copilot (Part 3\) \- Code Reviews & Hotfixes**

**Students arrive with:**

* Week 9 SDLC and TDD experience  
* Self-paced videos: Pull request best practices, code review strategies, hotfix workflows, Copilot for code review

**We teach/practice:**

* Creating effective pull requests  
* Using Copilot for code review and quality checks  
* Review strategies for ML code  
* Hotfix workflows with Copilot assistance  
* Emergency response scenarios

**Labs:**

1. Create PRs with proper documentation and Copilot-assisted descriptions  
2. Conduct code reviews using Copilot to identify issues, suggest improvements  
3. Practice hotfix scenarios (production bug, urgent fix needed) with proper gitflow  
4. Simulate emergency response with time pressure

**Skills developed:**

* Can create professional pull requests  
* Can conduct thorough code reviews with Copilot assistance  
* Can handle production hotfixes efficiently  
* Ready for collaborative, production ML development

**Extra/Optional (async):**

* Advanced PR strategies (draft PRs, review apps)  
* Automated code review tools  
* Post-mortem analysis practices  
* Incident response protocols

## **Week 11: Large Language Models (LLMs)**

**Students arrive with:**

* All previous ML/AWS/Git experience  
* Self-paced videos: LLM architectures (transformers, attention), Hugging Face ecosystem, tokenization concepts, API vs local deployment

**We teach/practice:**

* Hugging Face tokenizers and AutoClasses (AutoModel, AutoTokenizer, AutoModelForCausalLM)  
* Three deployment patterns:  
* Cloud APIs (OpenAI, Anthropic)  
* Local hosting (vLLM, Ollama)  
* Direct HF model loading in code  
* Prompting techniques (zero-shot, few-shot, chain-of-thought)  
* When to use each deployment option

**Labs:**

1. Hugging Face basics: Load tokenizer, tokenize text, use AutoClasses for inference  
2. Compare inference methods:  
* Call OpenAI/Anthropic APIs  
* Run local model with vLLM/Ollama  
* Load HF model directly in code  
1. Prompting workshop: Practice different prompting strategies for various tasks (classification, extraction, reasoning)  
2. Build simple prompt templates and evaluate responses

**Skills developed:**

* Can work with Hugging Face tokenizers and models  
* Understand trade-offs between cloud APIs, local hosting, and direct model loading  
* Can implement effective prompting strategies  
* Can choose appropriate LLM deployment for different use cases

**Extra/Optional (async):**

* Transformer architecture deep dive  
* Advanced tokenization internals  
* Model comparison (GPT-4, Claude, Llama)

## **Week 12: GenAI for Data Science**

**Students arrive with:**

* Week 11 LLM fundamentals, prompting experience  
* Self-paced videos: Synthetic data generation, LLM evaluation metrics, prompt engineering for data tasks, LLM-assisted analysis

**We teach/practice:**

* Using LLMs for data augmentation and synthetic data generation  
* Evaluating LLM-powered applications (accuracy, consistency, hallucinations)  
* Error analysis techniques for LLM outputs  
* Using LLMs for exploratory data analysis, visualization generation, and insight extraction

**Labs:**

1. Generate synthetic training data using LLMs (text classification, NER, etc.)  
2. Build evaluation framework for LLM outputs (automated metrics \+ human review)  
3. Error analysis: Categorize and analyze LLM failures, improve prompts iteratively  
4. LLM-assisted data analysis: Upload dataset, use LLM to generate plots, extract insights, suggest analysis approaches

**Skills developed:**

* Can use LLMs to augment training datasets  
* Can evaluate and measure LLM application quality  
* Can systematically analyze and improve LLM outputs  
* Can leverage LLMs for exploratory data analysis

**Extra/Optional (async):**

* Advanced synthetic data techniques  
* Custom evaluation metrics  
* Automated prompt optimization

## **Week 13: Amazon Bedrock**

**Students arrive with:**

* Week 11-12 LLM and GenAI experience  
* Self-paced videos: Amazon Bedrock overview, available models (Claude, Llama, Titan, etc.), Bedrock pricing, Knowledge Bases concepts

**We teach/practice:**

* Model selection criteria (performance, cost, latency, context length, capabilities)  
* Testing and comparing different Bedrock models for specific use cases  
* Pricing calculation and cost optimization strategies  
* Setting up and using Bedrock Knowledge Bases for RAG

**Labs:**

1. Model comparison: Test multiple Bedrock models (Claude, Llama, etc.) on same task, evaluate quality/speed/cost  
2. Pricing estimation exercise: Calculate costs for different scenarios (API calls, token usage, knowledge base queries)  
3. Build a Knowledge Base: Upload documents, configure chunking/embeddings, query with natural language  
4. Cost optimization: Implement strategies (prompt caching, model routing, batch processing)

**Skills developed:**

* Can select appropriate Bedrock models based on requirements  
* Can estimate and optimize costs for LLM applications  
* Can build and query Bedrock Knowledge Bases  
* Understand trade-offs in managed LLM services

**Extra/Optional (async):**

* Advanced Bedrock features  
* Guardrails implementation  
* Multi-model routing strategies

## **Week 14: Training AI Models**

**Students arrive with:**

* Week 11-13 LLM and Bedrock experience  
* Self-paced videos: Fine-tuning concepts, LoRA/QLoRA theory, parameter-efficient fine-tuning (PEFT), distillation basics

**We teach/practice:**

* Fine-tuning Hugging Face models with LoRA and QLoRA  
* When to use full fine-tuning vs LoRA vs QLoRA  
* Preparing datasets for fine-tuning  
* Evaluation of fine-tuned models  
* (Optional) OpenAI/Azure OpenAI fine-tuning API  
* (Time permitting) Model distillation from large Bedrock model to smaller open-source LLM

**Labs:**

1. Fine-tune a Hugging Face model using LoRA on custom dataset (classification or instruction-following)  
2. Compare LoRA vs QLoRA: memory usage, training time, performance  
3. Evaluate fine-tuned model against base model  
4. (Optional/Demo) Azure OpenAI fine-tuning walkthrough  
5. (Time permitting) Distill knowledge from Claude/Bedrock model to smaller open-source model

**Skills developed:**

* Can fine-tune LLMs using LoRA and QLoRA  
* Understand parameter-efficient fine-tuning techniques  
* Can prepare datasets and evaluate fine-tuned models  
* Know when to fine-tune vs prompt engineering

**Extra/Optional (async):**

* Full fine-tuning strategies  
* Advanced PEFT techniques  
* Model distillation deep dive

## **Week 15: Developing Agentic AI**

**Students arrive with:**

* Week 11-14 LLM and fine-tuning experience  
* Self-paced videos: Agent architectures, ReAct pattern (Reasoning \+ Acting), LangChain framework, memory systems, tool calling

**We teach/practice:**

* Building ReAct agents (observe, think, act loop)  
* LangChain fundamentals: chains, agents, tools  
* Implementing multi-step reasoning  
* Memory systems (conversation history, semantic memory, entity memory)  
* Tool integration (calculators, APIs, databases)

**Labs:**

1. Build a ReAct agent that can reason and use tools to solve problems  
2. Create LangChain agents with custom tools (web search, calculator, database query)  
3. Implement different memory types (conversation buffer, summary, vector store)  
4. Multi-step reasoning task: Agent that breaks down complex problems and solves step-by-step

**Skills developed:**

* Can build ReAct-style agents  
* Can use LangChain to create agents with tools  
* Can implement memory systems for stateful conversations  
* Understand multi-step reasoning patterns

**Extra/Optional (async):**

* Advanced agent architectures  
* Custom tool creation  
* Agent debugging strategies  
* Cost optimization for agents

## **Week 16: Integrating Agentic AI**

**Students arrive with:**

* Week 15 ReAct agents and LangChain experience  
* Self-paced videos: LangGraph for stateful workflows, multi-agent systems, agent communication patterns, orchestration strategies

**We teach/practice:**

* Building complex workflows with LangGraph (state machines, conditional routing)  
* Multi-agent systems: specialized agents working together  
* Agent orchestration and communication  
* Supervisor patterns and hierarchical agents

**Labs:**

1. Build LangGraph workflow with multiple states and conditional branching  
2. Create multi-agent system: multiple specialized agents collaborating on a task (e.g., researcher \+ writer \+ critic)  
3. Implement supervisor agent that coordinates worker agents  
4. Error handling and recovery in agent systems

**Skills developed:**

* Can build complex stateful workflows with LangGraph  
* Can design and implement multi-agent systems  
* Can orchestrate agent communication and collaboration  
* Understand agent system architectures

**Extra/Optional (async):**

* Human-in-the-loop workflows  
* Parallel agent execution  
* Advanced orchestration patterns

## **BREAK WEEK after Week 16**

## **Week 17: Retrieval-Augmented Generation (RAG) \- Part 1**

**Students arrive with:**

* Week 11-16 LLM and Agentic AI experience  
* Self-paced videos: RAG concepts, embedding models, vector databases, similarity search

**We teach/practice:**

* Understanding embedding models (OpenAI, Sentence Transformers, Cohere)  
* Vector databases (FAISS, Pinecone, Chroma)  
* Building basic RAG pipeline: embed → store → retrieve → generate  
* Semantic search fundamentals

**Labs:**

1. Generate embeddings with different models, compare quality  
2. Set up FAISS vector database, index documents  
3. Build end-to-end RAG pipeline: ingest documents → create embeddings → store in FAISS → query → retrieve relevant chunks → generate answer with LLM  
4. Compare retrieval quality with different embedding models

**Skills developed:**

* Can work with embedding models  
* Can set up and use vector databases (FAISS)  
* Can build basic RAG pipelines  
* Understand semantic search principles

**Extra/Optional (async):**

* Other vector databases (Pinecone, Weaviate, Qdrant)  
* Embedding dimensionality considerations  
* Hybrid search (dense \+ sparse)

## **Week 18: Retrieval-Augmented Generation (RAG) \- Part 2**

**Students arrive with:**

* Week 17 basic RAG pipeline experience  
* Self-paced videos: Chunking strategies, RAG evaluation metrics, reranking techniques, advanced retrieval

**We teach/practice:**

* Chunking strategies: fixed-size, semantic, recursive, their impact on retrieval quality  
* RAG evaluation: faithfulness, relevance, answer quality metrics  
* Reranking techniques to improve retrieval accuracy  
* Optimizing RAG pipelines

**Labs:**

1. Experiment with different chunking strategies on same corpus, measure impact on retrieval  
2. Implement RAG evaluation framework (RAGAS, custom metrics)  
3. Add reranking layer to RAG pipeline (Cohere rerank, cross-encoders)  
4. A/B test different RAG configurations, optimize for accuracy

**Skills developed:**

* Can implement and evaluate different chunking strategies  
* Can measure and improve RAG system quality  
* Can implement reranking for better retrieval  
* Can optimize end-to-end RAG pipelines

**Extra/Optional (async):**

* Advanced chunking strategies (sliding windows, overlapping)  
* Query expansion techniques  
* Contextual compression

## **Week 19: Machine Learning Operations (MLOps) \- Part 1**

**Students arrive with:**

* All previous ML, LLM, and RAG experience  
* Self-paced videos: DVC (Data Version Control), MLflow advanced features, experiment tracking best practices, model versioning

**We teach/practice:**

* Data versioning with DVC (track datasets, pipelines, reproducibility)  
* Model versioning in SageMaker Model Registry  
* MLflow deep dive: experiments, runs, parameters, metrics, artifacts, model registry  
* Reproducible experimentation workflows

**Labs:**

1. Set up DVC for data and model versioning, track changes across experiments  
2. Run multiple experiments with MLflow, log parameters/metrics/artifacts  
3. Compare experiments and select best model  
4. Register models in both MLflow and SageMaker registries  
5. Reproduce experiments from version control

**Skills developed:**

* Can version data and models with DVC  
* Can use MLflow for comprehensive experiment tracking  
* Can manage model versions across platforms  
* Understand reproducibility in ML workflows

**Extra/Optional (async):**

* DVC pipelines  
* Remote storage configuration (S3, GCS)  
* MLflow projects and models  
* Advanced versioning strategies

## **Week 20: Machine Learning Operations (MLOps) \- Part 2**

**Students arrive with:**

* Week 19 versioning and experimentation experience  
* Self-paced videos: CI/CD for ML, data drift detection, model monitoring, LLM observability tools (Langfuse, LiteLLM)

**We teach/practice:**

* Building CI/CD pipelines for ML models (GitHub Actions, automated testing, deployment)  
* Detecting data drift and concept drift  
* Setting up monitoring and alerting for production models  
* LLM-specific observability: tracking prompts, responses, costs, latency with Langfuse and LiteLLM

**Labs:**

1. Build CI/CD pipeline: code push → tests → model training → deployment  
2. Implement data drift detection on production data  
3. Set up model monitoring dashboards and alerts  
4. Integrate Langfuse for LLM observability: track all LLM calls, analyze costs, debug chains  
5. Use LiteLLM as unified interface with built-in logging

**Skills developed:**

* Can build CI/CD pipelines for ML systems  
* Can detect and respond to data/model drift  
* Can implement comprehensive monitoring and alerting  
* Can track and optimize LLM applications in production

**Extra/Optional (async):**

* Advanced CI/CD patterns (blue-green deployment, canary releases)  
* Custom drift detection algorithms  
* Cost optimization for LLM applications

## **Week 21: Apache Airflow \- Part 1 (Fundamentals)**

**Students arrive with:**

* Week 19-20 MLOps experience  
* Self-paced videos: Airflow architecture, DAG concepts, operators, task dependencies, scheduling

**We teach/practice:**

* Airflow fundamentals: DAGs, tasks, operators, dependencies  
* Setting up local Airflow environment  
* Building DAGs with branches and conditionals  
* Integrating Slack for alerting  
* ML workflows in Airflow (focus on orchestration)

**Labs:**

1. Set up local Airflow, explore UI  
2. Build simple ML pipeline DAG (data ingestion → preprocessing → training → evaluation)  
3. Implement branching logic and conditional execution (e.g., only deploy if accuracy \> threshold)  
4. Configure Slack notifications for task failures and successes  
5. Practice task dependencies and execution order

**Skills developed:**

* Can build and run Airflow DAGs locally  
* Can implement branching and conditional logic  
* Can set up alerting (Slack)  
* Understand Airflow orchestration for ML workflows

**Extra/Optional (async):**

* Task groups and SubDAGs  
* XCom for inter-task communication  
* Dynamic task generation

## **Week 22: Apache Airflow \- Part 2 (Production & Best Practices)**

**Students arrive with:**

* Week 21 local Airflow experience  
* Self-paced videos: Airflow executors (Celery, KEDA), remote deployment, dynamic DAG generation, error handling

**We teach/practice:**

* Deploying Airflow in remote environments (AWS, Kubernetes)  
* Celery Executor for distributed task execution  
* KEDA Executor for auto-scaling workloads  
* AutoDAG pattern: programmatically generating DAGs  
* Airflow best practices: retries, SLAs, idempotency, failure handling, backfilling

**Labs:**

1. Deploy Airflow to remote environment (MWAA on AWS or Airflow on K8s)  
2. Configure Celery executor for distributed execution  
3. Set up KEDA executor for auto-scaling based on queue depth  
4. Implement AutoDAG pattern: generate multiple similar DAGs programmatically  
5. Build resilient DAGs: retries, timeouts, failure callbacks, idempotent tasks  
6. Practice backfilling and catchup scenarios

**Skills developed:**

* Can deploy and manage Airflow in production  
* Understand different executors and when to use them  
* Can generate DAGs programmatically  
* Can build resilient, production-ready ML pipelines  
* Understand Airflow best practices for reliability

**Extra/Optional (async):**

* KEDA executor deep dive  
* Custom operators  
* Airflow plugins  
* Complex scheduling patterns (cron, timetables)

## **Week 23: AI Governance & Ethics**

**Students arrive with:**

* All previous ML/LLM/MLOps experience  
* Self-paced videos: AI ethics principles, bias in ML systems, fairness metrics, explainability concepts, regulatory landscape (GDPR, EU AI Act)

**We teach/practice:**

* Detecting and mitigating bias with Fairlearn and AI Fairness 360 (IBM)  
* Model explainability with SHAP and LIME  
* Microsoft Responsible AI Toolbox for comprehensive analysis  
* Regulatory overview: GDPR, EU AI Act, sector-specific requirements (discussion-based)  
* Building responsible AI practices into ML workflows

**Labs:**

1. Use Fairlearn to assess model fairness across demographic groups, apply mitigation techniques  
2. Generate SHAP explanations for model predictions, identify important features  
3. Use AI Fairness 360 to measure bias metrics and compare mitigation algorithms  
4. Run Microsoft Responsible AI Toolbox dashboard: error analysis, fairness assessment, explainability  
5. Case study analysis: Review real-world AI failures, discuss ethical implications

**Regulation Discussion (no hands-on lab):**

* GDPR compliance: right to explanation, data minimization  
* EU AI Act: risk categorization, prohibited practices, high-risk requirements  
* Industry-specific regulations (finance, healthcare, etc.)  
* Documentation and audit requirements

**Skills developed:**

* Can assess and mitigate bias in ML models  
* Can generate model explanations for stakeholders  
* Understand regulatory requirements for AI systems  
* Can integrate responsible AI practices into development workflow

**Extra/Optional (async):**

* Deep dive into regulations  
* Documentation templates  
* Fairness-accuracy tradeoffs

## **Week 24: Capstone Project**

**Labs:**

1. Students present their capstone projects (choose one):  
* Production RAG system with monitoring  
* Multi-agent research system  
* Complete MLOps pipeline with orchestration  
1. Live code walkthrough  
2. Architecture and design discussion  
3. Challenges faced and solutions implemented  
4. Peer feedback and Q\&A  
* 