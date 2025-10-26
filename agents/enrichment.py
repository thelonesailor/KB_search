from langgraph.graph import StateGraph, END
from typing import Dict, Any, Literal, Union
import json
import logging
from clients.perplexity_client import perplexity_client
from config import settings
from models import AgentState
from retrieval.core import QdrantRAGCore

logger = logging.getLogger(__name__)


def safe_get(obj: Union[dict, object], key: str, default=None):
    """Safely get value from dict or object attribute"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return getattr(obj, key, default)


class QueryAnalysisAgent:
    def __init__(self, model: str = None):
        self.model = model or settings.chat_model
        self.client = perplexity_client.get_client()

    def analyze_query(self, query: str):
        """Analyze user query using Perplexity"""

        system_prompt = """
        You are a query analysis expert. Analyze the user's query to determine:
        1. Primary intent (factual_lookup, analytical, procedural, ambiguous)
        2. Sub-questions that need to be answered
        3. Required data elements for a complete answer
        4. Confidence in understanding the query
        5. Whether web search augmentation might be helpful

        You MUST respond with ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
        {
            "intent": "factual_lookup|analytical|procedural|ambiguous",
            "sub_questions": ["question1", "question2"],
            "required_data_elements": ["element1", "element2"],
            "confidence": 0.85,
            "needs_web_search": false
        }
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response (handle markdown code blocks and <think> tags)
        try:
            # Remove <think>...</think> tags from reasoning models
            if '<think>' in response_text and '</think>' in response_text:
                # Extract content after </think>
                response_text = response_text.split('</think>')[-1].strip()

            # Remove markdown code blocks if present
            if '```json' in response_text:
                # Extract content between ```json and ```
                parts = response_text.split('```json')
                if len(parts) > 1:
                    response_text = parts[1].split('```')[0].strip()
            elif response_text.startswith("```"):
                # Handle generic code blocks
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Fallback to default structure if JSON parsing fails
            logger.warning(f"Failed to parse query analysis JSON: {e}. Using defaults.")
            logger.debug(f"Failed response text: {response_text[:500]}")  # Log first 500 chars only
            analysis_data = {
                "intent": "factual_lookup",
                "sub_questions": [],
                "required_data_elements": [],
                "confidence": 0.5,
                "needs_web_search": False
            }

        analysis_data["original_query"] = query  # Preserve original query

        return analysis_data


class ReflectionAgent:
    def __init__(self, model: str = None):
        self.model = model or settings.reasoning_model
        self.client = perplexity_client.get_client()

    def reflect_on_output(self, query: str, query_analysis, generation_output):
        """Reflect on output using Perplexity's reasoning models"""

        # Handle both dict and object access patterns
        if isinstance(generation_output, dict):
            answer = generation_output.get('answer', 'No answer')
            sources = generation_output.get('sources', [])
            confidence = generation_output.get('confidence', 0.0)
        else:
            answer = getattr(generation_output, 'answer', 'No answer')
            sources = getattr(generation_output, 'sources', [])
            confidence = getattr(generation_output, 'confidence', 0.0)

        # Handle query_analysis similarly
        if isinstance(query_analysis, dict):
            intent = query_analysis.get('intent', 'unknown')
            required_data = query_analysis.get('required_data_elements', [])
        else:
            intent = getattr(query_analysis, 'intent', 'unknown')
            required_data = getattr(query_analysis, 'required_data_elements', [])

        reflection_prompt = f"""
        Analyze the following Q&A interaction for completeness and clarity:

        ORIGINAL QUERY: {query}
        QUERY INTENT: {intent}
        REQUIRED DATA: {', '.join(required_data)}

        GENERATED ANSWER: {answer}
        SOURCES: {', '.join(sources)}
        CONFIDENCE: {confidence}

        CRITICAL INSTRUCTIONS:
        - Be PRACTICAL and USER-FOCUSED
        - If the answer directly addresses the user's question with factual information, mark it as COMPLETE
        - Only flag missing_elements if they are ESSENTIAL to answer the user's actual question
        - Do NOT request meta-information like "confidence justification" or "source reliability confirmation"
        - Focus on SUBSTANTIVE content gaps, not process/metadata gaps
        - If sources are cited in the answer text, consider them present even if the sources list is empty

        Assess:
        1. Does the answer provide what the user actually asked for?
        2. Are there CRITICAL information gaps that prevent answering the question?
        3. Is there ambiguity that makes the answer unusable?

        You MUST respond with ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
        {{
            "is_complete": true,
            "missing_elements": ["element1"],
            "ambiguity_detected": false,
            "clarifying_question": "question or null",
            "confidence_score": 0.85,
            "needs_web_search": false
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a reflection agent that evaluates answer completeness. Always respond with valid JSON only."},
                {"role": "user", "content": reflection_prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response (handle markdown code blocks and <think> tags)
        try:
            # Remove <think>...</think> tags from reasoning models
            if '<think>' in response_text and '</think>' in response_text:
                # Extract content after </think>
                response_text = response_text.split('</think>')[-1].strip()

            # Remove markdown code blocks if present
            if '```json' in response_text:
                # Extract content between ```json and ```
                parts = response_text.split('```json')
                if len(parts) > 1:
                    response_text = parts[1].split('```')[0].strip()
            elif '```' in response_text:
                # Handle generic code blocks
                parts = response_text.split('```')
                if len(parts) >= 3:
                    # Get content between first and second ```
                    response_text = parts[1].strip()
                    # Remove 'json' language identifier if present
                    if response_text.startswith('json'):
                        response_text = response_text[4:].strip()

            reflection_result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Fallback to default structure if JSON parsing fails
            logger.warning(f"Failed to parse reflection JSON: {e}. Using defaults.")
            logger.debug(f"Problematic response (first 500 chars): {response_text[:500]}")
            reflection_result = {
                "is_complete": True,
                "missing_elements": [],
                "ambiguity_detected": False,
                "clarifying_question": None,
                "confidence_score": 0.7,
                "needs_web_search": False
            }

        return reflection_result


class EnrichmentOrchestrator:
    def __init__(self, rag_core: QdrantRAGCore):
        self.rag_core = rag_core
        self.query_analyzer = QueryAnalysisAgent()
        self.reflection_agent = ReflectionAgent()

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("execute_rag", self._execute_rag_node)
        workflow.add_node("reflect_on_output", self._reflect_on_output_node)
        workflow.add_node("handle_ambiguity", self._handle_ambiguity_node)
        workflow.add_node("enrich_data", self._enrich_data_node)
        workflow.add_node("generate_final_answer", self._generate_final_answer_node)

        # Define edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "execute_rag")

        # Add conditional edge: skip reflection if RAG confidence is high
        workflow.add_conditional_edges(
            "execute_rag",
            self._should_reflect,
            {
                "reflect": "reflect_on_output",
                "skip": "generate_final_answer"
            }
        )

        # Conditional edges from reflection
        workflow.add_conditional_edges(
            "reflect_on_output",
            self._route_after_reflection,
            {
                "complete": "generate_final_answer",
                "ambiguous": "handle_ambiguity",
                "incomplete": "enrich_data",
                "retry": "execute_rag"
            }
        )

        workflow.add_edge("handle_ambiguity", "execute_rag")
        workflow.add_edge("enrich_data", "execute_rag")
        workflow.add_edge("generate_final_answer", END)

        # Compile with increased recursion limit
        return workflow.compile(checkpointer=None, debug=False)

    def _analyze_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Query Analysis & Decomposition"""
        state.execution_trace.append("Starting query analysis")

        query_analysis = self.query_analyzer.analyze_query(state.user_query)
        state.query_analysis = query_analysis

        intent = safe_get(query_analysis, 'intent', 'unknown')
        confidence = safe_get(query_analysis, 'confidence', 0.5)

        state.execution_trace.append(
            f"Query analyzed: intent={intent}, confidence={confidence}")

        return {"query_analysis": query_analysis}

    def _execute_rag_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Initial RAG Execution"""
        state.retry_count += 1
        state.execution_trace.append(f"Executing RAG core (attempt {state.retry_count})")

        # If we have clarification, enhance the query
        current_query = state.user_query
        if state.clarification_response:
            current_query = f"{state.user_query} [Clarification: {state.clarification_response}]"

        generation_output = self.rag_core.retrieve_and_generate(
            current_query,
            state.query_analysis
        )
        state.generation_output = generation_output

        confidence = safe_get(generation_output, 'confidence', 0.0)
        sources = safe_get(generation_output, 'sources', [])

        state.execution_trace.append(
            f"RAG executed: confidence={confidence}, sources={len(sources)}")

        return {"generation_output": generation_output, "retry_count": state.retry_count}

    def _reflect_on_output_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Reflection & Ambiguity Detection"""
        state.execution_trace.append("Reflecting on RAG output")

        reflection_result = self.reflection_agent.reflect_on_output(
            state.user_query,
            state.query_analysis,
            state.generation_output
        )
        state.reflection_result = reflection_result

        is_complete = safe_get(reflection_result, 'is_complete', True)
        ambiguity = safe_get(reflection_result, 'ambiguity_detected', False)
        missing = safe_get(reflection_result, 'missing_elements', [])

        state.execution_trace.append(
            f"Reflection complete: complete={is_complete}, "
            f"ambiguity={ambiguity}, "
            f"missing_elements={len(missing)}"
        )

        return {"reflection_result": reflection_result}

    def _handle_ambiguity_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Handle ambiguity by asking for clarification"""
        state.execution_trace.append("Handling ambiguity - clarification needed")

        # In a real implementation, this would pause execution and wait for human input
        # For MVP, we'll simulate this by setting a flag
        clarifying_question = safe_get(state.reflection_result, 'clarifying_question', 'Could you please provide more details?')

        state.execution_trace.append(f"Asking for clarification: {clarifying_question}")

        # For demo purposes, we'll simulate a clarification response
        # In production, this would come from user interaction
        simulated_response = "Please provide the most recent available data"
        state.clarification_response = simulated_response

        return {"clarification_response": simulated_response}

    def _enrich_data_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Dynamic Data Augmentation"""
        state.execution_trace.append("Initiating dynamic data enrichment")

        missing_elements = safe_get(state.reflection_result, 'missing_elements', [])

        # Simulate external data enrichment
        # In production, this would call actual APIs/tools
        enriched_data = {}
        for element in missing_elements[:2]:  # Limit to 2 elements for demo
            enriched_data[element] = f"Simulated data for {element} from external system"

        state.enriched_data = enriched_data
        state.execution_trace.append(f"Enriched data: {list(enriched_data.keys())}")

        return {"enriched_data": enriched_data}

    def _generate_final_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Generate final answer"""
        state.execution_trace.append("Generating final answer")

        # Use the RAG output as final answer, or enhance with enriched data
        final_answer = safe_get(state.generation_output, 'answer', 'No answer generated.')

        if state.enriched_data:
            enrichment_note = "\n\nAdditional enriched data:\n" + "\n".join(
                f"- {k}: {v}" for k, v in state.enriched_data.items()
            )
            final_answer += enrichment_note

        # Generate enrichment suggestions based on missing elements
        enrichment_suggestions = self._generate_enrichment_suggestions(state)
        state.enrichment_suggestions = enrichment_suggestions

        state.final_answer = final_answer
        state.execution_trace.append("Final answer generated successfully")

        return {"final_answer": final_answer, "enrichment_suggestions": enrichment_suggestions}

    def _generate_enrichment_suggestions(self, state: AgentState) -> list:
        """Generate actionable enrichment suggestions based on missing information"""
        suggestions = []

        reflection = state.reflection_result
        generation_output = state.generation_output

        if not reflection or not generation_output:
            return suggestions

        missing_elements = safe_get(reflection, 'missing_elements', [])
        is_complete = safe_get(reflection, 'is_complete', True)
        sources = safe_get(generation_output, 'sources', [])
        confidence = safe_get(generation_output, 'confidence', 0.0)
        answer = safe_get(generation_output, 'answer', '')

        # Filter out meta/process-related missing elements - focus on substantive content
        substantive_missing = [
            elem for elem in missing_elements
            if not any(meta_term in elem.lower() for meta_term in [
                'confidence', 'justification', 'reliability', 'confirmation',
                'verification_method', 'score', 'source_reliability'
            ])
        ]

        # Only suggest enrichment if truly incomplete
        if not is_complete and substantive_missing and confidence < 0.7:
            for element in substantive_missing[:3]:  # Limit to top 3
                # Make suggestions more user-friendly
                readable_element = element.replace('_', ' ').title()
                suggestion = {
                    "type": "missing_data",
                    "element": readable_element,
                    "action": f"Consider uploading documents with information about {readable_element}",
                    "priority": "medium"
                }
                suggestions.append(suggestion)

        # Check if answer has sources but they're not in the sources array
        has_source_citations = '[Source:' in answer or '[source:' in answer.lower()

        # Only flag no sources if truly no sources and low confidence
        if len(sources) == 0 and not has_source_citations and confidence < 0.5:
            suggestions.append({
                "type": "no_sources",
                "element": "Source Documents",
                "action": "Upload relevant documents to improve answer quality",
                "priority": "high"
            })
        elif confidence < 0.4 and len(sources) < 2:
            suggestions.append({
                "type": "low_confidence",
                "element": "Additional Context",
                "action": "Consider uploading more detailed documents for better accuracy",
                "priority": "medium"
            })

        return suggestions

    def _should_reflect(self, state: AgentState) -> Literal["reflect", "skip"]:
        """Decide if reflection is needed based on RAG confidence"""
        generation_output = state.generation_output

        if isinstance(generation_output, dict):
            confidence = generation_output.get('confidence', 0.0)
            sources = generation_output.get('sources', [])
        else:
            confidence = getattr(generation_output, 'confidence', 0.0)
            sources = getattr(generation_output, 'sources', [])

        # Skip reflection if we have high confidence and sources
        if confidence > 0.7 and len(sources) > 0:
            state.execution_trace.append(f"Skipping reflection (high confidence: {confidence})")
            return "skip"

        return "reflect"

    def _route_after_reflection(self, state: AgentState) -> Literal["complete", "ambiguous", "incomplete", "retry"]:
        """Conditional routing based on reflection results"""
        MAX_RETRIES = 1  # Maximum number of RAG execution attempts

        reflection = state.reflection_result

        is_complete = safe_get(reflection, 'is_complete', True)
        confidence_score = safe_get(reflection, 'confidence_score', 0.7)
        ambiguity_detected = safe_get(reflection, 'ambiguity_detected', False)
        missing_elements = safe_get(reflection, 'missing_elements', [])

        # Force completion if max retries reached
        if state.retry_count >= MAX_RETRIES:
            logger.warning(f"Max retries ({MAX_RETRIES}) reached. Forcing completion.")
            state.execution_trace.append(f"Forcing completion after {state.retry_count} attempts")
            return "complete"

        # Normal routing logic
        # Lowered threshold to 0.6 to avoid unnecessary retries for reasonable answers
        if is_complete and confidence_score > 0.6:
            return "complete"
        elif ambiguity_detected and state.retry_count < 2:
            # Only handle ambiguity on first retry
            return "ambiguous"
        elif not is_complete and missing_elements and state.retry_count < 2:
            # Only try enrichment on first retry
            return "incomplete"
        elif state.retry_count < MAX_RETRIES:
            return "retry"
        else:
            return "complete"

    def process_query(self, query: str) -> AgentState:
        """Main entry point for processing queries"""
        initial_state = AgentState(user_query=query, retry_count=0)

        # Execute the workflow with increased recursion limit
        final_state = self.workflow.invoke(
            initial_state,
            config={"recursion_limit": 50}  # Increased from default 25
        )

        return final_state