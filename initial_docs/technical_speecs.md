# **Technical Environment Specifications \- Bread Financial AI Academy**

## **Overview**

This document provides infrastructure-as-code style specifications for all technical environments required for the 24-week AI Academy curriculum.  
**Academy Timeline:** 24 weeks delivered over 12 months (2 weeks per month)  
**Session Duration:** 2 hours per week per cohort  
**Total Monthly Usage:** 6 hours of infrastructure time (2 hours × 3 cohorts) × 2 weeks \= 12 hours per cohort per month

## **1\. SageMaker Environment**

**Status:** ✅ Covered by Author  
**Infrastructure:** AWS PS Account with Terraform  
**Notes:** Author will provision and manage SageMaker resources using existing Terraform configurations.

## **2\. Airflow Environment**

**Status:** ✅ Covered by Author  
**Infrastructure:** AWS Managed Workflows for Apache Airflow (MWAA) via Terraform  
**Notes:** Author will set up remote Airflow environment on AWS using PS account.

## **3\. Azure Databricks Environment**

### **Student Organization**

* **Total Students:** 60  
* **Cohort Structure:** 3 cohorts of 20 students each  
* **Session Duration:** 2 hours per cohort  
* **Cluster Strategy:** 1 dedicated cluster per cohort with autoscaling

### **Workspace Configuration**

`resource "azurerm_databricks_workspace" "academy" {`  
  `name                = "databricks-academy-workspace"`  
  `resource_group_name = azurerm_resource_group.academy.name`  
  `location            = azurerm_resource_group.academy.location`  
  `sku                 = "standard"  # or "premium" if advanced features needed`  
`}`

### **ML Runtime Clusters (3 Cohort Clusters)**

`# Create 3 clusters - one per cohort`  
`resource "databricks_cluster" "cohort_clusters" {`  
  `for_each = toset(["cohort-1", "cohort-2", "cohort-3"])`

  `cluster_name            = "academy-ml-${each.key}"`  
  `spark_version           = "14.3.x-scala2.12"  # Latest LTS with ML Runtime`  
  `node_type_id            = "Standard_DS4_v2"   # 8 cores, 28GB RAM per node`  
  `driver_node_type_id     = "Standard_DS4_v2"   # Same as worker for consistency`

  `autoscale {`  
    `min_workers = 2   # Minimum capacity for responsive startup`  
    `max_workers = 8   # Handles 20 concurrent students comfortably`  
  `}`

  `# Auto-termination after session ends`  
  `autotermination_minutes = 30  # Terminates after 30 min of inactivity`

  `spark_conf = {`  
    `"spark.databricks.delta.preview.enabled" = "true"`  
    `"spark.sql.adaptive.enabled"              = "true"`  
  `}`

  `custom_tags = {`  
    `"Environment" = "Academy"`  
    `"Cohort"      = each.key`  
    `"Purpose"     = "ML-Training"`  
  `}`

  `# Enable cluster log delivery for troubleshooting`  
  `cluster_log_conf {`  
    `dbfs {`  
      `destination = "dbfs:/cluster-logs/${each.key}"`  
    `}`  
  `}`  
`}`

### **Cluster Sizing Rationale**

* **Instance Type:** Standard\_DS4\_v2 (8 cores, 28GB RAM)  
* **Min Workers:** 2 nodes \= 16 cores, 56GB RAM total (enough for basic operations)  
* **Max Workers:** 8 nodes \= 64 cores, 224GB RAM total (handles 20 students doing ML workloads)  
* **Autoscaling:** Responds to workload demand during 2-hour sessions  
* **Auto-termination:** 30 minutes saves costs between sessions

### **Required Libraries (Pre-installed in ML Runtime)**

* Spark MLlib (included)  
* pandas  
* numpy  
* scikit-learn  
* matplotlib  
* seaborn

### **Cluster Pool (Optional \- For Cost Optimization)**

`# Optional: Pre-warmed instances for faster cluster startup`  
`resource "azurerm_databricks_cluster_pool" "academy_pool" {`  
  `name                  = "academy-ml-pool"`  
  `min_idle_instances    = 3   # Keep 3 instances warm`  
  `max_capacity          = 24  # Support all 3 clusters at max scale`  
  `idle_instance_autotermination_minutes = 15`

  `instance_pool_spec {`  
    `node_type_id = "Standard_DS4_v2"`  
  `}`  
`}`

`# Reference pool in cluster configuration`  
`# Add to each cluster block:`  
`# instance_pool_id = azurerm_databricks_cluster_pool.academy_pool.id`

### **User Management**

#### **User Provisioning (60 Students)**

`# Create users for all 60 students`  
`resource "databricks_user" "students" {`  
  `for_each  = toset(var.student_emails)  # List of 60 email addresses`  
  `user_name = each.value`

  `workspace_access = true`  
  `databricks_sql_access = false  # Not needed for academy`  
`}`

`# Create cohort-specific groups`  
`resource "databricks_group" "cohort_1" {`  
  `display_name = "Academy-Cohort-1"`  
`}`

`resource "databricks_group" "cohort_2" {`  
  `display_name = "Academy-Cohort-2"`  
`}`

`resource "databricks_group" "cohort_3" {`  
  `display_name = "Academy-Cohort-3"`  
`}`

`# Master group for all students`  
`resource "databricks_group" "all_students" {`  
  `display_name = "Academy-All-Students"`  
`}`

`# Add users to their respective cohort groups`  
`# Cohort 1: Students 1-20`  
`resource "databricks_group_member" "cohort_1_members" {`  
  `for_each  = toset(slice(var.student_emails, 0, 20))`  
  `group_id  = databricks_group.cohort_1.id`  
  `member_id = databricks_user.students[each.key].id`  
`}`

`# Cohort 2: Students 21-40`  
`resource "databricks_group_member" "cohort_2_members" {`  
  `for_each  = toset(slice(var.student_emails, 20, 40))`  
  `group_id  = databricks_group.cohort_2.id`  
  `member_id = databricks_user.students[each.key].id`  
`}`

`# Cohort 3: Students 41-60`  
`resource "databricks_group_member" "cohort_3_members" {`  
  `for_each  = toset(slice(var.student_emails, 40, 60))`  
  `group_id  = databricks_group.cohort_3.id`  
  `member_id = databricks_user.students[each.key].id`  
`}`

`# Add all users to master group`  
`resource "databricks_group_member" "all_student_members" {`  
  `for_each  = toset(var.student_emails)`  
  `group_id  = databricks_group.all_students.id`  
  `member_id = databricks_user.students[each.key].id`  
`}`

#### **Cluster Access Permissions (Cohort-Specific)**

`# Cohort 1 - Access to cluster 1`  
`resource "databricks_permissions" "cohort_1_cluster" {`  
  `cluster_id = databricks_cluster.cohort_clusters["cohort-1"].id`

  `access_control {`  
    `group_name       = databricks_group.cohort_1.display_name`  
    `permission_level = "CAN_ATTACH_TO"`  
  `}`  
`}`

`# Cohort 2 - Access to cluster 2`  
`resource "databricks_permissions" "cohort_2_cluster" {`  
  `cluster_id = databricks_cluster.cohort_clusters["cohort-2"].id`

  `access_control {`  
    `group_name       = databricks_group.cohort_2.display_name`  
    `permission_level = "CAN_ATTACH_TO"`  
  `}`  
`}`

`# Cohort 3 - Access to cluster 3`  
`resource "databricks_permissions" "cohort_3_cluster" {`  
  `cluster_id = databricks_cluster.cohort_clusters["cohort-3"].id`

  `access_control {`  
    `group_name       = databricks_group.cohort_3.display_name`  
    `permission_level = "CAN_ATTACH_TO"`  
  `}`  
`}`

### **Storage Configuration**

`# Azure Storage for datasets`  
`resource "azurerm_storage_account" "academy_data" {`  
  `name                     = "academydatastorage"`  
  `resource_group_name      = azurerm_resource_group.academy.name`  
  `location                 = azurerm_resource_group.academy.location`  
  `account_tier             = "Standard"`  
  `account_replication_type = "LRS"`  
`}`

`resource "azurerm_storage_container" "datasets" {`  
  `name                  = "datasets"`  
  `storage_account_name  = azurerm_storage_account.academy_data.name`  
  `container_access_type = "private"`  
`}`

`# Mount in Databricks`  
`# Configure via Databricks Secret Scope and DBFS mount`

### **Cost Optimization Settings**

* **Auto-termination:** 30 minutes of inactivity  
* **Cluster Type:** Standard (not High-Concurrency, not GPU)  
* **Runtime:** ML Runtime (Standard, not GPU-enabled)  
* **Instance Type:** DS3\_v2 (cost-effective, sufficient for curriculum)  
* **Autoscaling:** Start at 1 worker, scale to 4 based on load

### **Required Variables**

`variable "student_emails" {`  
  `description = "List of all 60 student email addresses"`  
  `type        = list(string)`  
  `default     = []  # Populate with 60 student emails`

  `validation {`  
    `condition     = length(var.student_emails) == 60`  
    `error_message = "Must provide exactly 60 student email addresses"`  
  `}`  
`}`

`variable "resource_group_name" {`  
  `description = "Azure Resource Group name"`  
  `type        = string`  
  `default     = "rg-databricks-academy"`  
`}`

`variable "location" {`  
  `description = "Azure region"`  
  `type        = string`  
  `default     = "East US"  # Adjust as needed`  
`}`

`# Example student email list structure:`  
`# student_emails = [`  
`#   # Cohort 1 (Students 1-20)`  
`#   "student1@company.com",`  
`#   "student2@company.com",`  
`#   ...`  
`#   "student20@company.com",`  
`#   # Cohort 2 (Students 21-40)`  
`#   "student21@company.com",`  
`#   ...`  
`#   "student40@company.com",`  
`#   # Cohort 3 (Students 41-60)`  
`#   "student41@company.com",`  
`#   ...`  
`#   "student60@company.com"`  
`# ]`

## **4\. Colab / JupyterLab Environment**

### **Primary Option: Google Colab**

* **Access Method:** Students use personal or work Google accounts  
* **Cost:** Free tier sufficient for curriculum  
* **GPU Access:** Free T4 GPUs available  
* **No Infrastructure Required**

### **Fallback Option: Managed JupyterLab (if Colab Restricted)**

#### **Infrastructure Specifications for 60 Students**

**Design Philosophy:** CPU-only instances with ample resources. We won't be using large models that require GPUs \- focus is on learning HuggingFace APIs, small models, and frameworks.

##### **Azure Container Instance Approach**

`# JupyterLab per student - 60 instances total`  
`resource "azurerm_container_group" "jupyter_students" {`  
  `for_each            = toset(var.student_emails)  # 60 students`  
  `name                = "jupyter-${replace(each.value, "@", "-")}"`  
  `resource_group_name = azurerm_resource_group.academy.name`  
  `location            = azurerm_resource_group.academy.location`  
  `os_type             = "Linux"`

  `container {`  
    `name   = "jupyterlab"`  
    `image  = "academyregistry.azurecr.io/jupyterlab-huggingface:latest"`  
    `cpu    = "4"      # 4 vCPUs for comfortable model inference`  
    `memory = "16"     # 16GB RAM - sufficient for small models and datasets`

    `ports {`  
      `port     = 8888`  
      `protocol = "TCP"`  
    `}`

    `environment_variables = {`  
      `JUPYTER_ENABLE_LAB = "yes"`  
      `STUDENT_EMAIL      = each.value`  
    `}`

    `# Persistent storage per student`  
    `volume {`  
      `name       = "workspace"`  
      `mount_path = "/home/jovyan/work"`

      `azure_file {`  
        `share_name           = azurerm_storage_share.student_workspaces[each.key].name`  
        `storage_account_name = azurerm_storage_account.academy_storage.name`  
        `storage_account_key  = azurerm_storage_account.academy_storage.primary_access_key`  
      `}`  
    `}`  
  `}`

  `ip_address_type = "Public"`  
  `dns_name_label  = "jupyter-${replace(each.value, "@", "-")}"`

  `tags = {`  
    `Student = each.value`  
    `Purpose = "Academy-JupyterLab"`  
  `}`  
`}`

##### **Instance Sizing Rationale**

* **4 vCPUs:** Handles small model inference (BERT-base, DistilBERT, small LLaMA models)  
* **16GB RAM:** Sufficient for:  
* Loading models up to \~7B parameters in 4-bit quantization  
* Processing reasonable datasets (up to 1M rows)  
* Running multiple notebooks simultaneously  
* Caching tokenizers and embeddings  
* **No GPU:** Focus is on API usage (OpenAI, Anthropic, Bedrock) and small local models for learning  
* **Cost-effective:** Significantly cheaper than GPU instances while meeting all curriculum needs

##### **Docker Image Specification**

`# Dockerfile for JupyterLab with HuggingFace Environment`  
`FROM jupyter/scipy-notebook:latest`

`USER root`

`# Install system dependencies`  
`RUN apt-get update && apt-get install -y \`  
    `git \`  
    `wget \`  
    `curl \`  
    `build-essential \`  
    `&& rm -rf /var/lib/apt/lists/*`

`USER jovyan`

`# Install Python packages for HuggingFace and LLM work`  
`RUN pip install --no-cache-dir \`  
    `# Core ML libraries`  
    `torch==2.1.0 \`  
    `torchvision==0.16.0 \`  
    `torchaudio==2.1.0 \`  
    `# HuggingFace ecosystem`  
    `transformers==4.35.0 \`  
    `datasets==2.14.0 \`  
    `tokenizers==0.15.0 \`  
    `accelerate==0.24.0 \`  
    `# Vector databases and embeddings`  
    `sentence-transformers==2.2.2 \`  
    `faiss-cpu==1.7.4 \`  
    `chromadb==0.4.18 \`  
    `# LLM frameworks`  
    `langchain==0.1.0 \`  
    `langgraph==0.0.20 \`  
    `# API clients`  
    `openai==1.3.0 \`  
    `anthropic==0.8.0 \`  
    `# Utilities`  
    `python-dotenv==1.0.0 \`  
    `tenacity==8.2.3 \`  
    `tiktoken==0.5.2 \`  
    `# Data science essentials`  
    `pandas==2.1.3 \`  
    `numpy==1.26.2 \`  
    `matplotlib==3.8.2 \`  
    `seaborn==0.13.0 \`  
    `scikit-learn==1.3.2 \`  
    `# Monitoring and evaluation`  
    `ragas==0.1.0 \`  
    `langfuse==2.0.0`

`# Set working directory`  
`WORKDIR /home/jovyan/work`

`# Configure JupyterLab`  
`RUN jupyter lab build`

`# Expose JupyterLab port`  
`EXPOSE 8888`

`CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]`

##### **Azure Container Registry**

`resource "azurerm_container_registry" "academy" {`  
  `name                = "academyregistry"`  
  `resource_group_name = azurerm_resource_group.academy.name`  
  `location            = azurerm_resource_group.academy.location`  
  `sku                 = "Basic"`  
  `admin_enabled       = true`  
`}`

`# Build and push Docker image to registry`

##### **Storage for Student Workspaces**

`resource "azurerm_storage_account" "academy_storage" {`  
  `name                     = "academystorage"`  
  `resource_group_name      = azurerm_resource_group.academy.name`  
  `location                 = azurerm_resource_group.academy.location`  
  `account_tier             = "Standard"`  
  `account_replication_type = "LRS"`  
`}`

`resource "azurerm_storage_share" "student_workspaces" {`  
  `for_each             = toset(var.student_emails)`  
  `name                 = "workspace-${replace(each.value, "@", "-")}"`  
  `storage_account_name = azurerm_storage_account.academy_storage.name`  
  `quota                = 50  # 50GB per student`  
`}`

##### **Access Management**

`# Output URLs for student access`  
`output "jupyter_urls" {`  
  `value = {`  
    `for email, instance in azurerm_container_group.jupyter_students :`  
    `email => "http://${instance.fqdn}:8888"`  
  `}`  
  `description = "JupyterLab access URLs for each student"`  
`}`

### **JupyterLab Environment Packages Summary**

* **PyTorch:** Latest stable with CPU support  
* **HuggingFace:** transformers, datasets, tokenizers, accelerate  
* **Vector Stores:** FAISS, ChromaDB, sentence-transformers  
* **LLM Frameworks:** LangChain, LangGraph  
* **API Clients:** OpenAI, Anthropic  
* **ML Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
* **Monitoring:** Langfuse, RAGAS

### **Cost Optimization for JupyterLab (60 Students)**

**Usage Pattern:** 2 hours per week × 2 weeks per month \= 4 hours per month per student

* **Auto-start/stop:** Containers only run during class hours (4 hours/month per student)  
* **Instance Type:** 4 vCPU, 16GB RAM (sufficient for all course needs without GPU)  
* **No GPU Instances:** Not needed \- using cloud APIs and small models  
* **Shared Storage:** Azure Files for persistent workspace data (50GB per student)  
* **Estimated Monthly Cost:** \~$80-100/month for all 60 students (only for actual usage hours)

## **5\. Localhost \+ Copilot Environment**

### **Student Machine Requirements**

#### **Minimum Specifications**

* **Operating System:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)  
* **RAM:** 8GB minimum (16GB recommended)  
* **Storage:** 10GB free space  
* **Internet:** Stable broadband connection for Copilot

#### **Required Software**

##### **Python Environment**

`# Python 3.10 or 3.11 (3.12+ may have compatibility issues with some packages)`  
`python --version  # Should output Python 3.10.x or 3.11.x`

##### **Python Installation Instructions**

* **Windows:** Download from python.org or use Microsoft Store  
* **macOS:** Use Homebrew (brew install python@3.11) or python.org  
* **Linux:** Use system package manager (apt install python3.11)

##### **Package Manager**

`# Verify pip is installed`  
`pip --version`

`# Upgrade pip`  
`pip install --upgrade pip`

#### **GitHub Copilot License**

* **Required:** GitHub Copilot subscription (individual or enterprise)  
* **IDE Integration:** VS Code with Copilot extension OR JetBrains with Copilot plugin  
* **Access:** Students must have active Copilot license throughout weeks 8-10, 19-20

#### **Recommended IDE Setup**

`# Visual Studio Code (recommended)`  
`# Install extensions:`  
`# - GitHub Copilot`  
`# - GitHub Copilot Chat`  
`# - Python`  
`# - Pylance`  
`# - GitLens`

#### **Git Configuration**

`# Git installation required`  
`git --version  # Should be 2.30+`

`# Basic configuration`  
`git config --global user.name "Student Name"`  
`git config --global user.email "student@example.com"`

#### **Python Packages (Install as needed per week)**

`# Week 8-10: Git/SDLC work`  
`pip install pytest pytest-cov black flake8 mypy`

`# Week 19-20: MLOps work`  
`pip install mlflow dvc[s3] great-expectations`  
`pip install langfuse litellm`  
`pip install pandas numpy scikit-learn`

#### **Environment Variables Template**

`# Create .env file for API keys (weeks 19-20)`  
`OPENAI_API_KEY=your_key_here`  
`ANTHROPIC_API_KEY=your_key_here`  
`AWS_ACCESS_KEY_ID=your_key_here`  
`AWS_SECRET_ACCESS_KEY=your_key_here`  
`LANGFUSE_PUBLIC_KEY=your_key_here`  
`LANGFUSE_SECRET_KEY=your_key_here`

#### **Virtual Environment Setup (Recommended)**

`# Create virtual environment`  
`python -m venv academy-env`

`# Activate (Windows)`  
`academy-env\Scripts\activate`

`# Activate (macOS/Linux)`  
`source academy-env/bin/activate`

## **Environment Schedule Reference**

| Weeks | Environment | Infrastructure Owner   |
| :---- | :---- | :---- |
| 1-2 | Colab / JupyterLab | Requires GPU access |
| 3-4 | Azure Databricks | Terraform spec provided |
| 5-7 | AWS SageMaker | Author (Terraform) |
| 8-10 | Localhost \+ Copilot | Student machines |
| 11-18 | Colab / JupyterLab | Requires GPU access |
| 19-20 | Localhost \+ Copilot | Student machines |
| 21-22 | AWS Airflow (MWAA) | Author (Terraform) |
| 23 | Colab / JupyterLab | No GPU required |
| 24 | Mixed | Based on capstone choice |

## 

## **Provisioning Checklist**

### **Pre-Course Setup (Week 0\)**

* Collect all student email addresses  
* Verify Colab access OR provision JupyterLab instances  
* Provision Azure Databricks workspace and cluster  
* Create Databricks users and assign permissions  
* Author provisions SageMaker environment  
* Author provisions Airflow (MWAA) environment  
* Distribute Copilot licenses to students  
* Send students localhost setup instructions

### **Week 1 Readiness**

* All students can access Colab/JupyterLab  
* Test PyTorch and HuggingFace library access  
* Verify GPU allocation (if using JupyterLab)

### **Week 3 Readiness**

* Databricks workspace accessible  
* All students can log in  
* Cluster starts successfully  
* Test Spark DataFrame operations

### **Week 5 Readiness**

* SageMaker access confirmed  
* S3 buckets provisioned  
* IAM roles configured

### **Week 8 Readiness**

* Students have Python 3 installed locally  
* Copilot licenses activated  
* Git configured on all machines  
* VS Code or IDE with Copilot installed

### **Week 21 Readiness**

* Airflow MWAA environment accessible  
* Students can access Airflow UI  
* Test DAG deployment

## 

## **Cost Estimates (Approximate Monthly)**

### **Azure Databricks (60 Students, 3 Cohorts)**

**Usage Pattern:** 2 hours per cohort per week × 2 weeks per month \= 4 hours per cohort per month \= 12 hours total cluster time per month

* **Workspace:** Standard SKU \- \~$5/month base (minimal usage)  
* **Compute (3 Clusters):**  
* DS4\_v2 instances (8 cores, 28GB RAM) \- \~$0.60/hour per node  
* Each cluster: 1 driver \+ 2-8 workers (autoscaling) \= 3-9 nodes  
* Conservative (avg 5 nodes): 5 nodes × $0.60/hour × 4 hours/month \= $12 per cohort  
* Peak (9 nodes): 9 nodes × $0.60/hour × 4 hours/month \= $21.60 per cohort  
* **Total for 3 cohorts: $36-65/month**  
* **Storage:** \~$10/month for datasets and Unity Catalog metastore  
* **Unity Catalog:** Free (included in Databricks pricing)  
* **Access Connector:** \~$5/month  
* **Estimated Total:** $55-85/month for 60 students

### **JupyterLab (if Colab not available \- 60 Students)**

**Usage Pattern:** 2 hours per week × 2 weeks per month \= 4 hours per month per student

* **Container Instances:** 4 vCPU, 16GB RAM \- \~$0.30/hour per container  
* Cost per student: $0.30/hour × 4 hours/month \= $1.20/month  
* **Total for 60 students: $72/month**  
* **Storage:** \~$2/month total (50GB per student, minimal usage)  
* **Container Registry:** \~$5/month  
* **Estimated Total:** $80-100/month for 60 students (actual usage hours only)

## 