# Project Setup and Execution Guide

## Prerequisites
- Python 3.x installed
- Virtual environment tool (built-in with Python)

## Setup
1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

## General Workflow for Each Step

The project is organized into sequential steps (e.g., `1_create_dataset`, `2_query_generation`, etc.). For any given step folder (referred to as `<step_folder>`), follow this process:

1. **Install dependencies:**
   ```bash
   pip install -r <step_folder>/requirements.txt
   ```

2. **Configure environment:**
   Copy the example environment file to `.env` inside the step folder and update the variables:
   ```bash
   cp <step_folder>/.env.example <step_folder>/.env
   ```
   Edit `<step_folder>/.env` to set the appropriate values for your environment.

3. **Run the main script:**
   Ensure `DRY_RUN` is set to `false` in your configuration, then run:
   ```bash
   python <step_folder>/main.py
   ```

4. **Test run (Optional):**
   To perform a dry run test:
   ```bash
   python <step_folder>/dry_run.py
   ```

## Available Steps
- `1_create_dataset`: Creates the initial dataset.
- `2_query_generation`: Generates queries based on the dataset.
- `3_e1_to_e4`: Implements E1 to E4 experiments using a RAG pipeline with components for embeddings, knowledge base loading, vector storage, reranking, and output validation.
- `4_evaluation`: Evaluates the outputs from previous steps.
- *(Future steps will follow the same structure)*

## Important Notes
- Always run commands from the root directory of the project. Do not change directories into the step folders.
- Output files and logs will be generated in the root folder.
