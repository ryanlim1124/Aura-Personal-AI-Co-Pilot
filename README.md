# 🧠 Aura — Personal AI Co-Pilot

Aura is a private, multi-modal AI agent application with a clean web interface. Built using **Streamlit** and **LangChain**, it allows users to run specialized AI personas locally to assist with finance, coding, research, and career preparation.

## 🌟 What It Is
Aura bridges the gap between raw command-line AI agents and polished commercial products. It provides a sleek, responsive web UI for interacting with locally hosted Large Language Models (like Llama 3.1 via Ollama) or cloud providers (OpenAI, Gemini). 

Instead of a generic chatbot, Aura uses highly specific **Agent Modes** equipped with tailored tools to perform specialized tasks accurately and affordably.

## 💡 Practicality & How It Helps People
* **Privacy First:** By supporting local models (Ollama), users can analyze private financial documents, personal notes, or proprietary code without sending data to third-party servers.
* **Cost-Effective Tooling:** Aura uses focused agent modes (e.g., Finance, Dev, Tutor). By limiting the tools available to each mode, it reduces token usage, speeds up response times, and prevents AI hallucinations.
* **Specialized Assistance:**
  * **Finance Mode:** Pulls live stock data and financial news to create daily briefings and analyze market trends.
  * **Dev Mode:** Acts as a senior developer to review local workspace code and explain concepts.
  * **Career Coach:** Conducts mock interviews and provides actionable feedback on specific industry standards.
  * **Tutor Mode:** Breaks down complex topics with analogies, real-world examples, and interactive quizzes.

## 🚀 Features
- **Dynamic Web UI:** Built with Streamlit, featuring a clean native theme, streaming typewriter responses, and chat memory.
- **Multiple AI Personas:** Switch between General Assistant, Researcher, Finance Analyst, Dev Mentor, Career Coach, and Tutor.
- **Integrated Tool Calling:** Agents can execute Python tools to search the web, read Wikipedia, fetch stock snapshots, and read local documents.
- **Provider Agnostic:** Easily swap between Ollama (Local), Google Gemini, or OpenAI by changing a single variable in the `.env` file.

## 🛠️️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/aura-ai-agent.git](https://github.com/yourusername/aura-ai-agent.git)
   cd aura-ai-agent

