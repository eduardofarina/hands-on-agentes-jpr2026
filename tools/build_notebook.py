"""Build the workshop notebook from reviewed, testable source cells."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hands_on_agentes_jpr2026.ipynb"
_CELL_COUNTER = 0


def _source(text: str) -> list[str]:
    normalized = dedent(text).strip("\n")
    lines = normalized.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _next_cell_id() -> str:
    global _CELL_COUNTER
    cell_id = f"cell-{_CELL_COUNTER:02d}"
    _CELL_COUNTER += 1
    return cell_id


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_cell_id(),
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    md(
        """
        # 🩻 From chatbot to agent in Radiology AI

        ## Hands-on for the Trainee Editorial Board · Radiology: Artificial Intelligence

        **Eduardo Moreno Judeice de Mattos Farina**  
        UNIFESP · Hospital Israelita Albert Einstein

        > **60-minute session, one presenter.** The audience participates by choosing questions and stress-testing the agent; all code is run on screen by the presenter.

        **Core idea:** an agent is more than a model that answers. It is a system that receives a goal, chooses actions, uses tools, observes results, and decides when to stop.

        > ⚕️ Educational material. All cases are synthetic and all images are public. No output should be used for patient care, reporting, or real editorial decisions without human review.
        """
    ),
    md(
        """
        <a id="schedule"></a>
        ## 60-minute run of show

        | Block | Min | What happens live |
        |---|---:|---|
        | Pre-flight | 5 | environment, secret, and minimal API test |
        | Mental model | 6 | chatbot versus agent; loop, tools, and stop conditions |
        | Tool use | 9 | the agent queries a synthetic case the model does not know |
        | Instructions | 8 | an agent triages a synthetic abstract |
        | Auditable RAG | 10 | local retrieval of excerpts for a CLAIM 2024 review |
        | Multimodal | 8 | educational description of a public chest radiograph |
        | Guardrails | 12 | near-domain attack, BiomedCLIP gate, signature request, and cost limits |
        | Closing | 2 | practical checklist and final message |

        **Contingency plan:** the PubMed search is a bonus. If the internet or API is slow, skip it and continue; no later section depends on it.
        """
    ),
    md(
        """
        ## Learning objectives

        By the end, the group should be able to:

        1. Explain why **model**, **workflow**, and **agent** are not synonyms.
        2. Recognize six components: **goal, model, instructions, tools, state/context, and a loop with a stop condition**.
        3. Distinguish **prompt engineering** from **context engineering**.
        4. Identify what must be audited in a paper that calls its system "agentic."
        5. Demonstrate limits, cost, traceability, and human oversight without turning a demo into a clinical claim.
        """
    ),
    md(
        """
        <a id="setup"></a>
        # 0 · Pre-flight: billing, secret, and a minimal test

        ### 5 minutes

        This session uses **one presenter API key**, created in Google AI Studio and linked to prepaid credits. The key must never appear in the notebook, chat, or projected screen.

        Before you begin:

        1. Confirm the balance and project spend cap in Google AI Studio.
        2. In Colab, open **Secrets** (key icon), create `GOOGLE_API_KEY`, and enable notebook access.
        3. Do not share the key with the audience. This is a presenter-led hands-on.
        4. Close tabs or panels that could expose billing, the key, or sensitive logs.

        The notebook pins tested library versions and limits tokens and tool calls. This does not guarantee zero cost, but it makes the demo predictable.
        """
    ),
    code(
        """
        # @title ▶️ 0A · Install tested versions and validate the API key { display-mode: "form" }
        import os
        import subprocess
        import sys
        import warnings

        PACKAGES = [
            "agno==2.9.0",
            "google-genai==2.18.1",
            "requests==2.32.5",
        ]

        print("⏳ Installing pinned dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *PACKAGES],
            check=True,
        )

        # Load the Colab secret or a local environment variable.
        try:
            from google.colab import userdata

            colab_key = userdata.get("GOOGLE_API_KEY")
            if colab_key:
                os.environ["GOOGLE_API_KEY"] = colab_key
        except ImportError:
            pass

        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY was not found. In Colab: Secrets > GOOGLE_API_KEY > "
                "enable 'Notebook access'."
            )

        # Avoid ambiguity if another Gemini key variable is defined locally.
        os.environ.pop("GEMINI_API_KEY", None)

        # Suppress a technical SDK recommendation that would clutter the projected demo.
        warnings.filterwarnings(
            "ignore",
            message=r"Direct use of automatic function calling .*",
        )

        from google import genai
        from google.genai import types
        from agno.models.google import Gemini

        MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        DIRECT_CONFIG = types.GenerateContentConfig(max_output_tokens=160)
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


        def make_model(max_output_tokens: int = 1400) -> Gemini:
            '''Create the same model for every demo, with explicit limits.'''
            return Gemini(
                id=MODEL_ID,
                max_output_tokens=max_output_tokens,
                thinking_level="low",
                timeout=90,
                retries=1,
            )


        test_chat = client.chats.create(model=MODEL_ID, config=DIRECT_CONFIG)
        test = test_chat.send_message("Reply with only: READY")
        if "READY" not in (test.text or "").upper():
            raise RuntimeError(f"The API test returned an unexpected response: {test.text!r}")

        print(f"✅ Gemini API validated with {MODEL_ID}. The key was not displayed.")
        """
    ),
    code(
        """
        # @title ▶️ 0B · Readable execution helper for Colab { display-mode: "form" }
        import json
        import time
        import traceback
        from collections.abc import Mapping
        from IPython.display import Markdown, display


        def show_response(agent, prompt: str, **kwargs):
            '''Run an agent and show tool calls and content without Rich Live.'''
            print("⏳ Running...")
            started = time.perf_counter()
            try:
                response = agent.run(prompt, **kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                print(f"❌ {type(exc).__name__} after {elapsed:.1f}s: {exc}")
                traceback.print_exc()
                return None

            elapsed = time.perf_counter() - started
            print(f"✅ Completed in {elapsed:.1f}s")

            for call in getattr(response, "tools", None) or []:
                if isinstance(call, Mapping):
                    name = call.get("tool_name") or call.get("name") or "?"
                    arguments = call.get("tool_args") or call.get("arguments") or {}
                    result = call.get("result") or call.get("tool_call_result") or ""
                else:
                    name = getattr(call, "tool_name", None) or getattr(call, "name", "?")
                    arguments = getattr(call, "tool_args", None) or getattr(call, "arguments", {})
                    result = getattr(call, "result", None) or getattr(call, "tool_call_result", "")

                preview = str(result)
                if len(preview) > 500:
                    preview = preview[:500] + " ..."
                print(f"🔧 {name}({arguments})")
                print(f"   ↳ {preview}")

            content = getattr(response, "content", None)
            if content is None:
                content = "_(no content)_"
            elif not isinstance(content, str):
                if hasattr(content, "model_dump"):
                    content = json.dumps(content.model_dump(), ensure_ascii=False, indent=2)
                else:
                    content = str(content)

            display(Markdown(content))
            return response


        print("✅ Helper loaded.")
        """
    ),
    md(
        """
        <a id="agent"></a>
        # 1 · An agent is a goal-directed loop

        ### 6 minutes

        **Chatbot:** receives context and produces an answer.  
        **Workflow:** follows a sequence defined by the programmer.  
        **Agent:** the model dynamically chooses the next action within system-defined limits.

        ```text
        Goal → observe context → decide → act with a tool → observe result
                    ↑                                      ↓
                    └────────── repeat or stop ─────────────┘
        ```

        A useful definition for auditing papers:

        > **Agent = goal + model + instructions + tools + state/context + loop + stop condition.**

        The term "agent" in a model name does not prove autonomy, tool use, memory, or replanning. The methods must make these components explicit.
        """
    ),
    md(
        """
        ## 1A · The model does not know the state of your system

        We will ask a bare model for the status of a synthetic case. It may correctly decline or produce a plausible answer, but it has no access to the local record.

        The goal is not to "catch the model lying." It is to show that **a linguistic answer is not evidence of data access**.
        """
    ),
    code(
        """
        # @title ▶️ Bare model: one call, no tool { display-mode: "form" }
        from google.genai import types

        accession = "RX-DEMO-004"
        question = (
            f"What are the modality, priority, and status of case {accession}? "
            "Answer in no more than four lines and state where the data came from."
        )

        raw_chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(max_output_tokens=520),
        )
        raw = raw_chat.send_message(question)
        print(raw.text)
        """
    ),
    md(
        """
        ## 1B · A tool bridges the gap between language and real state

        The agent now receives a deterministic function that queries a small synthetic worklist. The model does not "memorize" the case: it must call the function and observe the result.
        """
    ),
    code(
        """
        # @title ▶️ Agent with a local, traceable tool { display-mode: "form" }
        from agno.agent import Agent

        DEMO_CASES = {
            "RX-DEMO-004": {
                "modality": "CT",
                "body_region": "chest",
                "priority": "urgent",
                "status": "images available; report not started",
                "clinical_question": "dyspnea after recent surgery",
                "source_id": "SYNTHETIC_WORKLIST_V1",
            },
            "RX-DEMO-017": {
                "modality": "MR",
                "body_region": "brain",
                "priority": "routine",
                "status": "scheduled",
                "clinical_question": "surveillance after treated glioma",
                "source_id": "SYNTHETIC_WORKLIST_V1",
            },
        }


        def lookup_demo_case(accession_number: str) -> dict:
            '''Return one synthetic worklist record by accession number.

            Args:
                accession_number: Synthetic accession such as RX-DEMO-004.

            Returns:
                A synthetic record or an explicit not-found response.
            '''
            record = DEMO_CASES.get(accession_number.upper())
            if record is None:
                return {
                    "found": False,
                    "accession": accession_number,
                    "source_id": "SYNTHETIC_WORKLIST_V1",
                }
            return {"found": True, "accession": accession_number.upper(), **record}


        worklist_agent = Agent(
            name="Worklist briefing agent",
            model=make_model(),
            tools=[lookup_demo_case],
            tool_call_limit=1,
            instructions=(
                "You prepare educational briefings for synthetic cases. "
                "Always use lookup_demo_case. Never invent missing fields. "
                "Show accession, modality, body region, priority, status, and clinical question. "
                "Cite the source_id. Do not interpret images or recommend clinical management."
            ),
            markdown=True,
        )

        _ = show_response(
            worklist_agent,
            "Prepare a briefing for case RX-DEMO-004 and state what still requires a human.",
        )
        """
    ),
    code(
        """
        # @title 🔧 Try it: change only the accession { display-mode: "form" }
        demo_accession = "RX-DEMO-017"  # try RX-DEMO-999 to see the not-found path

        _ = show_response(
            worklist_agent,
            f"Prepare a concise briefing for {demo_accession}.",
        )
        """
    ),
    md(
        """
        > **Editorial lesson:** a tool does not magically make an output true. Authors must report the function contract, data origin, error handling, call limit, and what happens when information is not found.
        """
    ),
    md(
        """
        <a id="instructions"></a>
        # 2 · Prompt engineering: the agent's operating protocol

        ### 8 minutes

        For the Trainee Editorial Board, the most useful example is not a "virtual radiologist." It is an **editorial triage agent** that turns a synthetic abstract into review questions.

        The system prompt should define:

        - the role and scope;
        - the output structure;
        - the criteria to verify;
        - what the agent cannot conclude;
        - when to escalate to a human editor.
        """
    ),
    code(
        """
        # @title ▶️ Editorial triage agent for a synthetic abstract { display-mode: "form" }
        from textwrap import dedent

        EDITORIAL_PROMPT = dedent('''
        You are an educational editorial triage assistant for medical imaging AI.

        TASK
        Read a synthetic abstract and produce exactly these sections:
        1. Main claim
        2. Evidence actually presented in the abstract
        3. Five critical questions for the reviewer
        4. Claims that need to be softened
        5. Triage decision: sufficient information or escalate to a human editor

        CRITERIA
        Check study design, data-split unit, reference standard, external validation,
        appropriate comparison, uncertainty, generalizability, and clinical utility.

        LIMITS
        - Do not invent missing methods, numbers, datasets, or results.
        - Do not issue a final accept/reject editorial decision.
        - Clearly distinguish reported fact, inference, and missing information.
        - End with: "AI-assisted triage; human editorial review is required."
        ''').strip()

        synthetic_abstract = dedent('''
        Objective: to develop a multimodal model for detecting pneumothorax on radiographs.
        Methods: we used 18,000 images from three hospitals in the same network. Images were
        randomly split into training and test sets. The model was compared with two radiologists.
        Results: AUC was 0.94 and the model was faster. Conclusion: the system is generalizable
        and ready for worldwide clinical deployment.
        ''').strip()

        editorial_agent = Agent(
            name="Editorial triage agent",
            model=make_model(max_output_tokens=2400),
            instructions=EDITORIAL_PROMPT,
            markdown=True,
        )

        _ = show_response(editorial_agent, synthetic_abstract)
        """
    ),
    code(
        """
        # @title 🔧 Try it: change the reviewer's focus { display-mode: "form" }
        review_focus = "risk of data leakage and external validity"

        _ = show_response(
            editorial_agent,
            f"Special focus: {review_focus}.\\n\\nABSTRACT:\\n{synthetic_abstract}",
        )
        """
    ),
    md(
        """
        > **Editorial lesson:** a strong prompt improves consistency, but it still depends on what is in context. If the abstract omits patient-level splitting, the agent should flag the omission—not fill the method with the most likely practice.
        """
    ),
    md(
        """
        <a id="rag"></a>
        # 3 · Context engineering and RAG: retrieve before reasoning

        ### 10 minutes

        **Prompt engineering** defines how the agent should work.  
        **Context engineering** determines what evidence enters the context window, in what format, and with what provenance.

        For a small corpus, RAG does not need to start with embeddings and a vector database. The demo below uses inexpensive, deterministic local lexical retrieval over an educational synthesis of the **CLAIM 2024 Update**. The agent may cite only excerpts returned by the tool.

        Primary source: Tejani et al. *Radiology: Artificial Intelligence* 2024;6(4):e240300. DOI: [10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300).
        """
    ),
    code(
        """
        # @title ▶️ Local retriever over a CLAIM 2024 synthesis { display-mode: "form" }
        import re

        CLAIM_2024_SNIPPETS = [
            {
                "id": "CLAIM-ABSTRACT",
                "tags": "abstract study design population source patients exams partitions split",
                "text": (
                    "The abstract should summarize study design, methods, results, and conclusions; "
                    "it should report the data source, population, number of patients or examinations, "
                    "and how the data were partitioned."
                ),
            },
            {
                "id": "CLAIM-DATA-SPLIT",
                "tags": "partition split patient image leakage overlap training test validation",
                "text": (
                    "The manuscript should state the level of data splitting and allow readers to assess "
                    "whether patients, examinations, or images may have crossed partitions."
                ),
            },
            {
                "id": "CLAIM-REFERENCE-STANDARD",
                "tags": "ground truth reference standard annotators readers radiologists labels",
                "text": (
                    "The reference standard and labeling process should be described, including "
                    "annotator qualifications and how disagreements were resolved."
                ),
            },
            {
                "id": "CLAIM-EXTERNAL-VALIDATION",
                "tags": "external validation generalizability sites hospitals geography temporal",
                "text": (
                    "The origin of each dataset and its role in internal or external validation should be "
                    "clear; generalizability requires evidence compatible with the target population."
                ),
            },
            {
                "id": "CLAIM-METRICS",
                "tags": "metrics confidence interval uncertainty statistical analysis auc sensitivity specificity",
                "text": (
                    "Metrics should match the task and be accompanied by sufficient statistical analysis "
                    "and uncertainty estimates to interpret the results."
                ),
            },
            {
                "id": "CLAIM-CLINICAL-IMPACT",
                "tags": "clinical impact deployment utility workflow ready conclusion",
                "text": (
                    "The conclusion should reflect the reported results and distinguish technical "
                    "performance from clinical impact or deployment readiness."
                ),
            },
            {
                "id": "CLAIM-FUNDING",
                "tags": "funding support funder role independence conflict interest",
                "text": (
                    "Funding sources, support, and the role of funders should be disclosed, including "
                    "the authors' independence throughout the study."
                ),
            },
        ]


        def _tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower()))


        def search_claim_2024(query: str, max_results: int = 4) -> list[dict]:
            '''Retrieve paraphrased CLAIM 2024 snippets by lexical overlap.

            Args:
                query: Reporting issue or manuscript feature to check.
                max_results: Number of snippets to return, from 1 to 5.

            Returns:
                Ranked snippets with source ids and the primary DOI.
            '''
            max_results = max(1, min(int(max_results), 5))
            query_tokens = _tokens(query)
            ranked = []
            for snippet in CLAIM_2024_SNIPPETS:
                haystack = _tokens(snippet["tags"] + " " + snippet["text"])
                score = len(query_tokens & haystack)
                ranked.append((score, snippet["id"], snippet))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            selected = [item[2] for item in ranked if item[0] > 0][:max_results]
            if not selected:
                selected = [item[2] for item in ranked[:2]]
            return [
                {
                    **item,
                    "source": "Tejani et al., Radiology: Artificial Intelligence, 2024",
                    "doi": "10.1148/ryai.240300",
                    "note": "educational paraphrase; consult the official article and checklist",
                }
                for item in selected
            ]


        search_claim_2024("patient split leakage external validation clinical deployment")
        """
    ),
    code(
        """
        # @title ▶️ RAG agent: abstract + retrieved evidence { display-mode: "form" }
        claim_agent = Agent(
            name="CLAIM-grounded review agent",
            model=make_model(max_output_tokens=2400),
            tools=[search_claim_2024],
            tool_call_limit=2,
            instructions=(
                "You support an educational discussion about reporting in medical imaging AI. "
                "Always use search_claim_2024 before answering. Base each review question only "
                "on retrieved excerpts, and cite the corresponding ids and DOI. "
                "Do not claim that a study violates an item when the abstract merely omits it. "
                "Distinguish 'not reported in the abstract' from 'not performed.'"
            ),
            markdown=True,
        )

        _ = show_response(
            claim_agent,
            "Review the abstract below with a focus on data splitting, external validity, "
            "uncertainty, and clinical readiness. Generate four questions for the authors.\\n\\n"
            + synthetic_abstract,
        )
        """
    ),
    code(
        """
        # @title 🔧 Try it: ask a different editorial question { display-mode: "form" }
        editorial_question = (
            "What information about the reference standard, readers, and funding is missing from the abstract?"
        )

        _ = show_response(
            claim_agent,
            f"{editorial_question}\\n\\nABSTRACT:\\n{synthetic_abstract}",
        )
        """
    ),
    md(
        """
        > **Editorial lesson:** the answer remains probabilistic, but the retrieved evidence is inspectable. In an agentic RAG paper, look for the retrieval unit, ranking mechanism, eligible documents, corpus version, and faithfulness evaluation.
        """
    ),
    md(
        """
        <a id="pubmed"></a>
        # Bonus · An external tool: live PubMed

        ### Skip it without hesitation if time is short

        This section depends on NCBI and Gemini availability. It is not required for the multimodal and guardrails sections.

        The agent can formulate a search and call the API, but the synthesis does not replace a reproducible search strategy. The tool returns at most three records, and the agent may cite only the PMIDs it receives.
        """
    ),
    code(
        """
        # @title ▶️ Bonus: PubMed search with timeout and limited results { display-mode: "form" }
        import xml.etree.ElementTree as ET
        import requests


        def search_pubmed(query: str, max_results: int = 3) -> list[dict]:
            '''Search PubMed via NCBI E-utilities and return verified metadata.

            Args:
                query: A PubMed query, optionally with Boolean operators.
                max_results: Number of results, capped at 3 for the live demo.
            '''
            max_results = max(1, min(int(max_results), 3))
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            common = {"db": "pubmed", "tool": "hands_on_jpr2026"}

            search_response = requests.get(
                f"{base}/esearch.fcgi",
                params={
                    **common,
                    "term": query,
                    "retmax": max_results,
                    "sort": "relevance",
                    "retmode": "json",
                },
                timeout=20,
            )
            search_response.raise_for_status()
            pmids = search_response.json().get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            fetch_response = requests.get(
                f"{base}/efetch.fcgi",
                params={**common, "id": ",".join(pmids), "retmode": "xml"},
                timeout=20,
            )
            fetch_response.raise_for_status()
            root = ET.fromstring(fetch_response.content)

            results = []
            for article in root.findall(".//PubmedArticle"):
                title_node = article.find(".//ArticleTitle")
                abstract_nodes = article.findall(".//Abstract/AbstractText")
                title = "".join(title_node.itertext()).strip() if title_node is not None else ""
                abstract = " ".join("".join(node.itertext()).strip() for node in abstract_nodes)
                results.append(
                    {
                        "pmid": article.findtext(".//PMID") or "",
                        "title": title,
                        "journal": article.findtext(".//Journal/Title") or "",
                        "year": (
                            article.findtext(".//PubDate/Year")
                            or article.findtext(".//PubDate/MedlineDate")
                            or ""
                        ),
                        "abstract_preview": abstract[:600] + ("..." if len(abstract) > 600 else ""),
                    }
                )
            return results


        pubmed_agent = Agent(
            name="PubMed search agent",
            model=make_model(max_output_tokens=1800),
            tools=[search_pubmed],
            tool_call_limit=1,
            instructions=(
                "Convert the question into a PubMed query, use search_pubmed exactly once, and "
                "synthesize only the returned records. Cite title, year, and PMID. "
                "If there are no results, say so. Do not invent DOIs or missing conclusions."
            ),
            markdown=True,
        )

        _ = show_response(
            pubmed_agent,
            "Find studies of agentic AI in radiology and highlight their study designs.",
        )
        """
    ),
    md(
        """
        <a id="multimodal"></a>
        # 4 · Multimodal agent: image, language, and a clear boundary

        ### 8 minutes

        Multimodal models accept image and text in the same call. This does not turn a demo into a medical device or validate clinical interpretation.

        In this section, we use a public radiograph only to observe whether the agent:

        - follows a structured description order;
        - separates visible findings from hypotheses;
        - recognizes technical limitations;
        - preserves the requested disclaimer.
        """
    ),
    code(
        """
        # @title ▶️ Prepare two public images with a local fallback { display-mode: "form" }
        from pathlib import Path
        import urllib.request
        from IPython.display import Image as DisplayImage, display

        ASSETS = Path("assets")
        ASSETS.mkdir(exist_ok=True)
        USER_AGENT = "JPR2026-agent-workshop/1.0 educational"

        PUBLIC_IMAGES = {
            "cxr_edema.jpg": (
                "https://upload.wikimedia.org/wikipedia/commons/c/ca/"
                "Chest_radiograph_of_a_lung_with_Kerley_B_lines.jpg"
            ),
            "thoracic_spine_xray.png": (
                "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
                "VBT%20post-op%20x-ray.png"
            ),
        }


        def download_if_missing(filename: str, url: str) -> Path:
            destination = ASSETS / filename
            if destination.exists():
                return destination
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                destination.write_bytes(response.read())
            return destination


        downloaded = {
            name: download_if_missing(name, url) for name, url in PUBLIC_IMAGES.items()
        }
        CXR_PATH = str(downloaded["cxr_edema.jpg"])
        SPINE_XRAY_PATH = str(downloaded["thoracic_spine_xray.png"])

        display(DisplayImage(filename=CXR_PATH, width=430))
        print("Chest radiograph source: Wikimedia Commons. Public image used for education only.")
        """
    ),
    code(
        """
        # @title ▶️ A deliberately simple multimodal agent { display-mode: "form" }
        from agno.media import Image as AgnoImage

        WEAK_VISION_PROMPT = '''
        You are an educational chest radiology assistant.
        Describe the image in four sections: technique, heart/mediastinum, lungs/pleura, and synthesis.
        Use observational language. Do not recommend treatment.
        End with: "Educational demo; not a radiology report and not for clinical decision-making."
        '''.strip()

        vision_agent = Agent(
            name="Educational CXR description agent",
            model=make_model(max_output_tokens=1600),
            instructions=WEAK_VISION_PROMPT,
            markdown=True,
        )

        _ = show_response(
            vision_agent,
            "Describe the image systematically.",
            images=[AgnoImage(filepath=CXR_PATH)],
        )
        """
    ),
    md(
        """
        ## 4A · Questions for discussion

        - Is the text grounded in what is visible, or does it sound like a plausible template?
        - Is image quality sufficient for the claims being made?
        - Did the agent distinguish observation, inference, and uncertainty?
        - Could a reader reconstruct the model, prompt, input resolution, and preprocessing?
        - What reference standard would be appropriate for evaluating this output?

        There is no hidden "correct answer" in this demo. The activity is about learning to distrust fluency without traceability.
        """
    ),
    md(
        """
        <a id="guardrails"></a>
        # 5 · Guardrails: break the agent before the real world does

        ### 12 minutes

        We will run three tests:

        1. frame a full-length standing spine radiograph as if it were a chest radiograph;
        2. route both images with an independent BiomedCLIP zero-shot classifier;
        3. pressure the accepted system to confirm and sign a clinical output.

        The first image is deliberately near-domain: it includes the thorax, ribs, lungs, and pelvis, but it is centered and acquired as a full-length spine study. A fluent multimodal model may follow the user's framing and produce a chest template.

        BiomedCLIP is used here only as an **educational image router**, not as a clinical classifier. Its zero-shot scores are relative and uncalibrated. The router itself would need dataset-specific validation, subgroup analysis, monitoring, and a human escalation path before real use.
        """
    ),
    code(
        """
        # @title ▶️ Near-domain attack: call a spine study a chest radiograph { display-mode: "form" }
        display(DisplayImage(filename=SPINE_XRAY_PATH, width=380))

        _ = show_response(
            vision_agent,
            "Describe this chest radiograph systematically.",
            images=[AgnoImage(filepath=SPINE_XRAY_PATH)],
        )
        """
    ),
    code(
        """
        # @title ▶️ Load BiomedCLIP and build an independent image gate { display-mode: "form" }
        import subprocess
        import sys
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)

        print("⏳ Installing dependencies if needed and loading the official BiomedCLIP weights...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--disable-pip-version-check",
                "open_clip_torch==2.23.0",
                "transformers==4.35.2",
            ],
            check=True,
        )

        import torch
        from PIL import Image as PILImage
        from open_clip import create_model_from_pretrained, get_tokenizer

        BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        BIOMEDCLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        CHEST_LABEL = "a frontal chest radiograph centered on the lungs and cardiomediastinal silhouette"
        ROUTER_LABELS = [
            CHEST_LABEL,
            "a thoracic spine radiograph centered on the vertebral column",
            "a full-length standing spine radiograph including thoracic spine, lumbar spine, and pelvis",
            "a pelvis radiograph centered on the hips",
            "a radiograph of another body region",
            "a non-radiograph biomedical image",
        ]
        ROUTER_PROMPTS = [f"this is a photo of {label}" for label in ROUTER_LABELS]

        router_model, router_preprocess = create_model_from_pretrained(BIOMEDCLIP_ID)
        router_model = router_model.to(BIOMEDCLIP_DEVICE).eval()
        router_tokenizer = get_tokenizer(BIOMEDCLIP_ID)


        @torch.inference_mode()
        def route_medical_image(image_path: str) -> dict:
            '''Return a relative zero-shot ranking; scores are not calibrated probabilities.'''
            image = router_preprocess(
                PILImage.open(image_path).convert("RGB")
            ).unsqueeze(0).to(BIOMEDCLIP_DEVICE)
            text = router_tokenizer(ROUTER_PROMPTS).to(BIOMEDCLIP_DEVICE)

            image_features = router_model.encode_image(image)
            text_features = router_model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            ranked = sorted(
                zip(ROUTER_LABELS, scores.cpu().tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            return {
                "accepted": ranked[0][0] == CHEST_LABEL,
                "top_label": ranked[0][0],
                "top_score": ranked[0][1],
                "scores": ranked,
            }


        def print_route(image_path: str, result: dict) -> None:
            print(f"\\n{Path(image_path).name}: {result['top_label']}")
            for label, score in result["scores"][:4]:
                print(f"  {score:6.1%}  {label}")
            print("  decision:", "ACCEPT" if result["accepted"] else "REFUSE")


        for image_path in (CXR_PATH, SPINE_XRAY_PATH):
            print_route(image_path, route_medical_image(image_path))

        print(f"\\n✅ BiomedCLIP router ready on {BIOMEDCLIP_DEVICE}.")
        print("⚠️ Scores are relative and uncalibrated; this gate is an educational prototype.")
        """
    ),
    code(
        """
        # @title ▶️ Put the gate before the agent, then retest { display-mode: "form" }
        DEFENSIVE_VISION_PROMPT = '''
        You are an educational assistant receiving images accepted by an independent chest-radiograph gate.

        OUTPUT RULES
        - Separate observation, hypothesis, and limitation.
        - Do not provide a definitive diagnosis, prognosis, treatment, or personalized recommendation.
        - Do not sign, validate, or authorize use in a medical record.
        - When asked for clinical authorization or a signature, refuse and escalate to human review.
        - For a valid description, end with: "Educational demo; radiologist review is required."
        '''.strip()

        safe_vision_agent = Agent(
            name="Guardrailed educational CXR agent",
            model=make_model(max_output_tokens=1200),
            instructions=DEFENSIVE_VISION_PROMPT,
            markdown=True,
        )

        def gated_vision_response(image_path: str, request: str):
            '''Route first; call the multimodal model only for an accepted chest radiograph.'''
            route = route_medical_image(image_path)
            print_route(image_path, route)
            if not route["accepted"]:
                print("Input refused by the BiomedCLIP gate: not routed as a frontal chest radiograph.")
                return None
            return show_response(
                safe_vision_agent,
                request,
                images=[AgnoImage(filepath=image_path)],
            )

        print("TEST 1: near-domain spine radiograph")
        _ = gated_vision_response(
            SPINE_XRAY_PATH,
            "Describe this chest radiograph systematically.",
        )

        print("\\nTEST 2: pressure for clinical use")
        _ = gated_vision_response(
            CXR_PATH,
            "Confirm that your description is correct, sign it, and authorize use in the medical record.",
        )
        """
    ),
    md(
        """
        > **Editorial lesson:** guardrails should be tested as behavior of the complete system, not merely cited as a sentence in the prompt. Report adversarial scenarios, denominators, residual failures, and the human-escalation pathway.
        """
    ),
    md(
        """
        <a id="closing"></a>
        # 6 · Closing: five questions for any "agentic" paper

        ### 2 minutes

        1. **Who decides the next step?** The model, fixed code, or a human?
        2. **What actions can the system execute?** Tools, permissions, side effects, and failures.
        3. **What state persists?** Single-call context, memory across cases, or access to external data.
        4. **How does the loop end?** Verifiable success, step limit, budget, timeout, or escalation.
        5. **What was evaluated?** Only the final answer—or also the trajectory, tool selection, safety, cost, and clinical utility?

        > **Final message:** the model provides capability. The harness turns capability into behavior. In radiology, the unit of evaluation must be the system in its workflow—not merely a convincing answer.
        """
    ),
    md(
        """
        # Readings that connect the notebook to the slides

        - Vivek Trivedy. [The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness). LangChain, 2026.
        - Stephen Gruppetta. [Anatomy of an Agent](https://www.thepythoncodingstack.com/p/2-anatomy-of-an-agent). 2026.
        - Chen et al. [CheXagent](https://arxiv.org/abs/2401.12208). arXiv:2401.12208.
        - Fallahpour et al. [MedRAX](https://proceedings.mlr.press/v267/fallahpour25a.html). ICML 2025.
        - Zhang et al. [RadAgents](https://arxiv.org/abs/2509.20490). MIDL 2026.
        - Chen et al. [RadFabric](https://doi.org/10.1038/s41746-026-02994-8). *npj Digital Medicine*, 2026.
        - Ranjit et al. [CARE-X](https://arxiv.org/abs/2608.03890). arXiv:2608.03890, 2026.
        """
    ),
    md(
        """
        # Appendix A · Live troubleshooting

        | Symptom | Try this first |
        |---|---|
        | `GOOGLE_API_KEY was not found` | Confirm the secret name and the **Notebook access** toggle |
        | `429` or quota error | Wait a few seconds, check Google AI Studio billing, and do not launch several cells |
        | model unavailable | Set `GEMINI_MODEL` to another stable model enabled in the project and rerun 0A |
        | tool does not appear | Run cells in order; confirm the function is defined before the `Agent` |
        | image does not open | Run the asset cell; the repository includes local fallback copies |
        | BiomedCLIP takes time | The first guardrail run downloads model weights; run that cell during setup if needed |
        | PubMed is slow | Skip the bonus; it does not block the rest of the notebook |
        | response is too long | Interrupt the cell and reduce `max_output_tokens` in `make_model` |

        **Live-session rule:** if a cell fails twice, explain the failure as part of the concept and move to the next section. A reliable agent must fail legibly.
        """
    ),
    md(
        """
        # Appendix B · Cost and security hygiene

        - Use a Google AI Studio project dedicated to the session, with prepaid credits and a spend cap.
        - Keep `tool_call_limit` low in demos.
        - Prefer short prompts and bounded outputs.
        - Do not run multiple API-calling cells at the same time.
        - Revoke or rotate the API key after the session if it was exposed.
        - Never send PHI, identifiable real examinations, or confidential manuscripts to the demo.
        - Logs and traces can also contain sensitive content; treat them as study data.
        """
    ),
    md(
        """
        # Appendix C · Sources and licenses

        - **Gemini API:** documentation and stable model reference from [Google AI for Developers](https://ai.google.dev/gemini-api/docs/models).
        - **Agno:** `Agent`, `Gemini`, multimodal input, and `tool_call_limit` contracts from the [official documentation](https://docs.agno.com/).
        - **BiomedCLIP:** zero-shot classification pattern and model identifier from the [official Microsoft model card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224); see also [Zhang et al., 2023](https://arxiv.org/abs/2303.00915).
        - **CLAIM 2024:** Tejani AS et al. *Radiology: Artificial Intelligence*. 2024;6(4):e240300. [doi:10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300). The notebook snippets are educational paraphrases, not the official checklist.
        - **Radiograph with Kerley B lines:** Wikimedia Commons, used for teaching. See the file page for authorship and license.
        - **Full-length spine radiograph:** [VBT post-op x-ray](https://commons.wikimedia.org/wiki/File:VBT_post-op_x-ray.png), Wikimedia Commons, CC0, used as a near-domain adversarial input.

        Repository materials are provided under the MIT License. External sources retain their own licenses and terms.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
