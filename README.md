# Hands-on: Agentes de IA para Radiologistas — JPR 2026

> Sessão prática de 1h-1h30 na **Jornada Paulista de Radiologia 2026** para radiologistas curiosos sobre como construir agentes de IA.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduardofarina/hands-on-agentes-jpr2026/blob/main/hands_on_agentes_jpr2026.ipynb)

## 🎯 O que você vai aprender

Ao final da sessão, você será capaz de:

1. **Explicar** a diferença entre um chatbot e um agente de IA.
2. **Identificar** os 4 componentes essenciais de um agente: modelo, instruções, ferramentas (*tools*) e memória/contexto.
3. **Distinguir** *prompt engineering* de *context engineering*.
4. **Reconhecer** quando vale a pena usar um agente — e quando um simples prompt resolve.
5. **Modificar** um agente pronto (trocar instruções, adicionar uma ferramenta) sem quebrar nada.
6. **Enxergar os limites**: alucinação, custo, latência, e por que *nenhum* agente deste notebook substitui um laudo.

## 🚀 Como rodar

1. **Antes da sessão** (3 minutos, faça em casa):
   - Acesse <https://console.groq.com/keys> e faça login (Google, GitHub ou e-mail).
   - Clique em **Create API Key** → dê um nome qualquer → copie a chave (começa com `gsk_...`).
   - Guarde — vamos usar no dia. Sem cartão de crédito.

2. **No dia da sessão**:
   - Clique no botão **"Open in Colab"** acima.
   - No menu esquerdo do Colab, clique no ícone 🔑 (*Secrets*).
   - Adicione um secret com:
     - **Nome**: `GROQ_API_KEY`
     - **Valor**: sua chave
     - Ative **Notebook access**.
   - Rode a célula de setup (a primeira com ▶️) e siga o notebook.

> ⚠️ **Nunca cole sua API key diretamente no código.** Sempre use o sistema de Secrets do Colab. Se você compartilhar o notebook com a chave dentro, qualquer pessoa pode usá-la.

> 🔄 **Plano B — Google Gemini:** se a Groq estiver fora do ar no dia, dá pra rodar o mesmo notebook trocando o provedor. Detalhes no apêndice B do próprio notebook.

## 📦 O que está neste repo

```
hands-on-agentes-jpr2026/
├── hands_on_agentes_jpr2026.ipynb   # O notebook da sessão (tudo está aqui)
├── assets/                          # Imagens de exemplo (radiografias públicas)
├── data/                            # PDFs de diretrizes para o RAG
├── requirements.txt                 # Dependências (Colab instala automaticamente)
├── LICENSE                          # MIT
└── README.md                        # Este arquivo
```

## 🧰 Stack utilizada

- **[Agno](https://github.com/agno-agi/agno)** — framework minimalista para construir agentes
- **[Groq Cloud](https://console.groq.com)** — inferência rápida (LPU) de modelos open-weight, com free tier sem cartão
  - **Llama 3.3 70B** (`llama-3.3-70b-versatile`) — texto e tool calling
  - **Llama 4 Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`) — multimodal (texto + imagem)
- **[Sentence-Transformers](https://www.sbert.net/)** — embeddings multilíngues locais para o RAG (sem API key)
- **Google Colab** — ambiente de execução gratuito, sem instalação local

## 📚 Pré-requisitos

- **Zero**. Sério. Se você nunca abriu um Jupyter notebook, a primeira seção ensina como clicar no ▶️.
- Conta Google (para o Colab) e uma conta na [Groq](https://console.groq.com) (Google/GitHub/e-mail, sem cartão).

## ⚕️ Aviso clínico

Este material é **exclusivamente educacional**. Nenhum dos agentes apresentados aqui é um dispositivo médico, não foi validado para uso clínico, e **não deve ser usado para tomar decisões sobre pacientes reais**. Os exemplos usam dados sintéticos ou imagens públicas de datasets abertos.

## 👤 Autor

**Eduardo Moreno Judeice de Mattos Farina**
UNIFESP · Hospital Israelita Albert Einstein

## 📄 Licença

[MIT](LICENSE) — sinta-se livre para adaptar, remixar e reusar em outras aulas, desde que mantenha a atribuição.

---

*Material preparado para a Jornada Paulista de Radiologia 2026.*
