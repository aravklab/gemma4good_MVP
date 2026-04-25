# GemmaGenius: System Architecture & Rules
This is a local, privacy-first educational CLI app. It uses the Feynman Technique (the student teaches the AI). 

## Core Rules
1. **No Agent Frameworks:** DO NOT use LangChain, LangGraph, AutoGen, or OpenAI libraries. 
2. **Tech Stack:** Raw Python 3.x only. Use the `requests` library to communicate with Ollama at `http://localhost:11434/api/generate`.
3. **State Management:** Use procedural while-loops and simple variables.
4. **The Models:** We use two distinct logical units hitting the same local model (e.g., `gemma2:2b`):
   - **Pip (The Actor):** An 8-year-old persona that requires dynamic prompt compilation.
   - **The Evaluator (The Logic):** A background process that strictly outputs JSON `{"classification": "..."}`.

## The Knowledge Graph
We use a hardcoded JSON dictionary for concepts to avoid vector DB overhead.