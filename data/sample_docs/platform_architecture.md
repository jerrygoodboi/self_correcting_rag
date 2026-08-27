# Intelligent Operations Management Platform (IOMP) Architecture

## Overview
The Intelligent Operations Management Platform (IOMP) is an enterprise-grade AI operations platform built for self-healing systems, automated root-cause analysis, and autonomous workflow execution.

## Core Components

### 1. Agentic Workflow Engine
The workflow engine is powered by LangGraph, executing cyclic, stateful workflows that enable multi-agent decision making. It features:
- State persistence backed by PostgreSQL checkpoints
- Fine-grained deterministic human-in-the-loop overrides
- Real-time event streaming and execution tracing

### 2. Knowledge Retrieval (Self-Correcting RAG)
The Self-Correcting RAG system implements active retrieval verification:
- Query understanding and intelligent routing
- Semantic similarity retrieval using vector embeddings
- LLM-based document relevance filtering
- Autonomous query reformulations and self-correction loops when retrieval confidence is insufficient
- Hallucination detection to guarantee that responses are 100% grounded in retrieved facts

### 3. Incident Auto-Remediation (Self-Healing)
Automated playbooks trigger when telemetry alerts fire:
- Level 1: Metric drift diagnosis
- Level 2: Log correlation & stack trace anomaly extraction
- Level 3: Safe automated rollback or container restarting with guardrails
