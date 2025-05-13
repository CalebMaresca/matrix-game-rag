---
title: Matrix Wargame RAG Agent
emoji: 🏃
colorFrom: pink
colorTo: gray
sdk: docker
pinned: false
license: agpl-3.0
short_description: Agentic RAG for designing/asking q abt matrix games
---

# Matrix Wargame RAG Agent - Certification Challenge

This repository contains the work for the Certification Challenge, focusing on an application to assist with understanding and designing matrix wargames.

## Introduction

This project aims to develop an AI-powered application that helps users answer questions about matrix wargames and supports them in the design process of new matrix wargames. It leverages Retrieval Augmented Generation (RAG) and agentic capabilities to provide accurate and contextually relevant information.

## Task 1: Defining your Problem and Audience

**Problem Statement:**
*   Users, such as game designers, researchers, and hobbyists, often find it challenging to quickly access specific information about matrix wargame mechanics, historical examples, and design principles, or need assistance in brainstorming and structuring new game designs.

**Why this is a problem for your specific user:**
*   While matrix games are designed to be accessible, newcomers can find it challenging to navigate the rules and gameplay without a knowledgeable facilitator or by frequently pausing to consult rulebooks. This application provides a centralized, interactive knowledge base to answer their specific questions in an accessible way.
*   Game designers may find it time consuming to build new games and struggle to incorporate all the necessary details and nuances. This application's game designer tool can help them create detailed and well-structured games more efficiently.
*   Researchers might need a tool to quickly compare and contrast different wargame designs or identify trends in the field.

**Potential questions that your user is likely to ask:**
*   "What are the core mechanics of a matrix wargame?"
*   "Can you give me examples of matrix wargames used for training military personnel?"
*   "How can I design a matrix wargame to simulate [specific scenario, e.g., a cybersecurity incident]?"
*   "What are common pitfalls to avoid when designing a matrix wargame?"
*   "Suggest some adjudication mechanisms for a diplomatic conflict in a matrix game."
*   "What data sources can I use to inform the design of a historical matrix wargame?"
*   "How do I balance a matrix wargame for multiple players with different objectives?"

## Task 2: Propose a Solution

**Proposed Solution:**
*   I will build an agentic RAG application that allows users to ask natural language questions about matrix wargames. The system will retrieve relevant information from a curated knowledge base of documents, articles, and potentially game rulebooks. It will also leverage agentic reasoning to help users brainstorm design elements, structure game phases, and consider different mechanics.
*   The user will interact with a simple interface (e.g., a web chat) where they can input their queries. The application will provide concise answers, cite sources, and offer follow-up suggestions or design considerations. This will save users time, provide targeted information, and act as a creative partner in the design process.

**Tooling Choices:**
*   **LLM:** GPT4.1-mini - Why: Newer model with a good balance between intelligence and cost
*   **Embedding Model:** snowflake-arctic-embed-l - Why: Open source and easy to fine tune
*   **Orchestration:** LangChain/LangGraph - Why: Easy to connect agent with many tools. Enables easy integration with more complex multi-agent graphs that I want to build in the future.
*   **Vector Database:** Qdrant - Why: Efficient similarity search and ease of integration with LangChain/LangGraph
*   **Monitoring:** LangSmith - Why:  To track performance, identify issues, and understand application usage
*   **Evaluation:** RAGAS- Why: To build custom personas of users and evaluate RAG metrics
*   **User Interface:** Chainlit - Why: For rapid prototyping. Currently building a Next.js frontend
*   **(Optional) Serving & Inference:** HF Spaces - Why: ease of implementation and access

**Agent Usage:**
*   An agent will be used to understand complex user queries that may require multi-step reasoning or access to multiple tools (e.g., a RAG retriever for the knowledge base and a wikipedia search tool for information on recent/historical events or broader context).
*   The Game designer tool enables the agent to design high-quality matrix games within a standardized Pydantic format (future features will allow users to save these games and submit them to be played by AI agents)

## Task 3: Dealing with the Data

**Data Sources and External APIs:**
*   **Primary Data Source:** A collection of PDFs, articles, and web content related to matrix wargaming theory, design principles, and existing game examples.
*   **External API:** Wikipedia API to allow the agent to look up information on events or other information it might need to design a game about a specific topic.

**Default Chunking Strategy:**
*   RecursiveCharacterTextSplitter with a chunk size of 300 characters
*   **Why:** Smaller chucks reduce token usage while maintaining high functionality (according to tests below)

## Task 4: Building a Quick End-to-End Prototype

**Deliverables:**
*   Link to deployed prototype: [Link to Hugging Face Space or other endpoint]

## Task 5: Creating a Golden Test Data Set

See `ragas_create_testset.ipynb`

**Deliverables:**
*   **RAGAS Evaluation Results:**
    | Metric             | Score |
    |--------------------|-------|
    | Context Recall         | 0.9361  |
    | Faithfulness           | 0.9439  |
    | Factual Correctness    | 0.7825  |
    | Answer Relevance       | 0.8933  |
    | Context Entity Recall  | 0.1368  |
    | Noise Sensitivity      | 0.1838  |

*   **Conclusions on Performance:**
    *   The first four scores look good, not sure why the last two are so low. Though, as discussed in class, these numbers don't have much meaning in isolation. Below, in Task 7, I will use them to compare performance before/after fine-tuning the embeddings.

## Task 6: Fine-Tuning Open-Source Embeddings

See `embeddings_fine_tune.ipynb`

**Deliverables:**
*   Link to fine-tuned embedding model on Hugging Face Hub: [Link to model]

## Task 7: Assessing Performance

**Deliverables:**
*   **Performance Comparison (RAGAS):**
    | Metric             | Original RAG Score | Fine-Tuned RAG Score |
    |--------------------|--------------------|------------------------|
    | Context Recall         | 0.9361            | 0.9681                 |
    | Faithfulness           | 0.9439            | 0.9744                 |
    | Factual Correctness    | 0.7825            | 0.8032                 |
    | Answer Relevance       | 0.8933             | 0.9221                 |
    | Context Entity Recall  | 0.1368             | 0.1456                |
    | Noise Sensitivity      | 0.1838             | 0.1580                |

    *   Performance increased on all metrics except for noise sensitivity!

*   **Changes for Second Half of Course:**
    *   I hope to develop this into a platform that people can use to play matrix games against AI agent players, or even simulate games entirly played by AI. This will allow for simulating many wargames in a short amount of time to explore possible futures and prepare planned actions to take in the most likely or most important contingencies.

## Links

*   **Loom Video (Demo & Use Case - 5 mins MAX):** [Link to Loom video]
*   **Public Application Link:** [Link to final Hugging Face Space or other deployment]
*   **Public Fine-Tuned Embedding Model Link:** [Link to Hugging Face Hub model]

