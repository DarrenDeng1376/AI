"""
Question-answering engine using Azure OpenAI and RAG (Retrieval-Augmented Generation)
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from openai import AzureOpenAI

from config import azure_config, qa_config
from .embedding_manager import EmbeddingManager, SearchResult
from .utils.azure_clients import AzureClientManager

logger = logging.getLogger(__name__)

class AnswerConfidence(Enum):
    """Confidence levels for answers"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class Answer:
    """Structured answer with metadata"""
    question: str
    answer: str
    confidence: AnswerConfidence
    confidence_score: float
    sources: List[SearchResult]
    context_used: str
    processing_time: float
    token_usage: Dict[str, int]
    follow_up_questions: List[str]
    reasoning: Optional[str] = None

@dataclass
class QASession:
    """Question-answering session to maintain context"""
    session_id: str
    conversation_history: List[Dict[str, str]]
    document_filter: Optional[str]
    created_at: float
    last_activity: float

class QAEngine:
    """Advanced question-answering engine with RAG capabilities"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize the QA engine"""
        self.embedding_manager = embedding_manager
        
        # Initialize Azure OpenAI client through client manager
        try:
            self.client_manager = AzureClientManager()
            self.openai_client = self.client_manager.get_openai_client()
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
        
        # Session management
        self.sessions: Dict[str, QASession] = {}
    
    async def answer_question(
        self,
        question: str,
        session_id: Optional[str] = None,
        document_filter: Optional[str] = None,
        use_conversation_context: bool = True
    ) -> Answer:
        """
        Answer a question using RAG with document context
        
        Args:
            question: The question to answer
            session_id: Optional session ID for conversation context
            document_filter: Filter to specific document
            use_conversation_context: Whether to use conversation history
            
        Returns:
            Answer object with response and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Get or create session
            session = None
            if session_id:
                session = self._get_or_create_session(session_id, document_filter)
            
            # Enhance question with conversation context if available
            enhanced_question = question
            if session and use_conversation_context:
                enhanced_question = self._enhance_question_with_context(question, session)
            
            # Retrieve relevant context
            search_results = await self.embedding_manager.hybrid_search(
                query=enhanced_question,
                max_results=qa_config.max_context_chunks,
                document_filter=document_filter
            )
            
            # Filter results by relevance threshold
            relevant_results = [
                result for result in search_results
                if result.similarity_score >= qa_config.context_relevance_threshold
            ]
            
            if not relevant_results:
                return self._create_no_context_answer(question, start_time)
            
            # Build context from relevant chunks
            context = self._build_context(relevant_results)
            
            # Generate answer using Azure OpenAI
            answer_response = await self._generate_answer(
                question=question,
                context=context,
                conversation_history=session.conversation_history if session else []
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                answer_response, relevant_results, question
            )
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(
                question, answer_response, context
            )
            
            # Update session if exists
            if session:
                self._update_session(session, question, answer_response)
            
            processing_time = time.time() - start_time
            
            answer = Answer(
                question=question,
                answer=answer_response["content"],
                confidence=confidence_level,
                confidence_score=confidence_score,
                sources=relevant_results,
                context_used=context,
                processing_time=processing_time,
                token_usage=answer_response.get("token_usage", {}),
                follow_up_questions=follow_up_questions,
                reasoning=answer_response.get("reasoning")
            )
            
            logger.info(f"Generated answer with {confidence_level.value} confidence in {processing_time:.2f}s")
            return answer
            
        except Exception as e:
            error_msg = f"Failed to answer question: {str(e)}"
            logger.error(error_msg)
            
            return Answer(
                question=question,
                answer=f"I encountered an error while processing your question: {error_msg}",
                confidence=AnswerConfidence.VERY_LOW,
                confidence_score=0.0,
                sources=[],
                context_used="",
                processing_time=time.time() - start_time,
                token_usage={},
                follow_up_questions=[],
                reasoning="Error occurred during processing"
            )
    
    async def _generate_answer(
        self,
        question: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate answer using Azure OpenAI"""
        try:
            # Build conversation messages
            messages = self._build_messages(question, context, conversation_history)
            
            # Call Azure OpenAI
            response = self.openai_client.chat.completions.create(
                model=azure_config.openai_deployment_name,
                messages=messages,
                temperature=qa_config.temperature,
                max_tokens=qa_config.max_tokens,
                top_p=qa_config.top_p
            )
            
            answer_content = response.choices[0].message.content
            
            # Extract reasoning if available (could be improved with structured outputs)
            reasoning = self._extract_reasoning(answer_content)
            
            return {
                "content": answer_content,
                "reasoning": reasoning,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise
    
    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Build conversation messages for OpenAI API"""
        messages = []
        
        # System prompt
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if available
        if conversation_history:
            for entry in conversation_history[-qa_config.max_follow_up_questions:]:
                messages.append({"role": "user", "content": entry["question"]})
                messages.append({"role": "assistant", "content": entry["answer"]})
        
        # Build context-aware user message
        user_message = self._build_user_message(question, context)
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI assistant"""
        return """You are an intelligent document analysis assistant. Your task is to answer questions based on the provided document context.

Guidelines:
1. Answer questions accurately based only on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Provide specific citations when possible (page numbers, document sections)
4. Be concise but thorough in your explanations
5. If you're uncertain about an answer, express your confidence level
6. For complex questions, break down your reasoning
7. Suggest related questions when appropriate

Always structure your response to be helpful and informative while staying grounded in the provided context."""
    
    def _build_user_message(self, question: str, context: str) -> str:
        """Build the user message with context"""
        return f"""Based on the following document context, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question, please state this clearly."""
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results):
            source_info = f"Source {i+1}"
            if result.document_name:
                source_info += f" ({result.document_name}"
                if result.page_number:
                    source_info += f", Page {result.page_number}"
                source_info += ")"
            
            context_part = f"{source_info}:\n{result.content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(
        self,
        answer_response: Dict[str, Any],
        search_results: List[SearchResult],
        question: str
    ) -> float:
        """Calculate confidence score for the answer"""
        confidence_factors = []
        
        # Factor 1: Quality of retrieved context
        if search_results:
            avg_similarity = sum(r.similarity_score for r in search_results) / len(search_results)
            confidence_factors.append(avg_similarity)
        else:
            confidence_factors.append(0.0)
        
        # Factor 2: Number of relevant sources
        source_factor = min(len(search_results) / qa_config.max_context_chunks, 1.0)
        confidence_factors.append(source_factor)
        
        # Factor 3: Answer length (too short might indicate lack of information)
        answer_length = len(answer_response.get("content", ""))
        length_factor = min(answer_length / 200, 1.0)  # Normalize to 200 chars
        confidence_factors.append(length_factor)
        
        # Factor 4: Presence of qualifying language (reduces confidence)
        answer_text = answer_response.get("content", "").lower()
        uncertainty_phrases = [
            "i don't know", "unclear", "uncertain", "might be", "possibly",
            "not enough information", "cannot determine", "insufficient context"
        ]
        
        uncertainty_factor = 1.0
        for phrase in uncertainty_phrases:
            if phrase in answer_text:
                uncertainty_factor -= 0.2
        
        uncertainty_factor = max(uncertainty_factor, 0.0)
        confidence_factors.append(uncertainty_factor)
        
        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Prioritize context quality
        weighted_confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return max(0.0, min(1.0, weighted_confidence))
    
    def _get_confidence_level(self, confidence_score: float) -> AnswerConfidence:
        """Convert confidence score to confidence level"""
        if confidence_score >= 0.8:
            return AnswerConfidence.HIGH
        elif confidence_score >= 0.6:
            return AnswerConfidence.MEDIUM
        elif confidence_score >= 0.4:
            return AnswerConfidence.LOW
        else:
            return AnswerConfidence.VERY_LOW
    
    def _extract_reasoning(self, answer_content: str) -> Optional[str]:
        """Extract reasoning from answer content (simple implementation)"""
        # Look for reasoning patterns in the answer
        reasoning_indicators = [
            "because", "since", "due to", "as a result", "therefore",
            "this is based on", "according to", "the context shows"
        ]
        
        sentences = answer_content.split('.')
        reasoning_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in reasoning_indicators):
                reasoning_sentences.append(sentence.strip())
        
        return '. '.join(reasoning_sentences) if reasoning_sentences else None
    
    async def _generate_follow_up_questions(
        self,
        original_question: str,
        answer: str,
        context: str
    ) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            follow_up_prompt = f"""Based on the following question and answer, suggest 2-3 relevant follow-up questions that could help the user explore the topic further.

Original Question: {original_question}
Answer: {answer}

Please provide follow-up questions that are:
1. Relevant to the topic
2. Likely to have answers in the same document context
3. Help deepen understanding of the subject

Format your response as a simple list of questions, one per line."""

            response = self.openai_client.chat.completions.create(
                model=azure_config.openai_deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions."},
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            follow_up_text = response.choices[0].message.content
            
            # Parse follow-up questions
            questions = []
            for line in follow_up_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Follow-up') and '?' in line:
                    # Clean up the question
                    question = line.split('?')[0] + '?'
                    question = question.strip('- ').strip('1. ').strip('2. ').strip('3. ')
                    if question:
                        questions.append(question)
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return []
    
    def _get_or_create_session(self, session_id: str, document_filter: Optional[str]) -> QASession:
        """Get existing session or create new one"""
        current_time = time.time()
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = current_time
            return session
        
        # Create new session
        session = QASession(
            session_id=session_id,
            conversation_history=[],
            document_filter=document_filter,
            created_at=current_time,
            last_activity=current_time
        )
        
        self.sessions[session_id] = session
        return session
    
    def _enhance_question_with_context(self, question: str, session: QASession) -> str:
        """Enhance question with conversation context"""
        if not session.conversation_history:
            return question
        
        # Simple context enhancement - could be improved
        recent_history = session.conversation_history[-2:]  # Last 2 exchanges
        
        if recent_history:
            context_summary = "Previous context: "
            for entry in recent_history:
                context_summary += f"Q: {entry['question'][:50]}... A: {entry['answer'][:50]}... "
            
            return f"{context_summary}\n\nCurrent question: {question}"
        
        return question
    
    def _update_session(self, session: QASession, question: str, answer: str) -> None:
        """Update session with new Q&A pair"""
        session.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Limit history size
        max_history = qa_config.max_follow_up_questions * 2
        if len(session.conversation_history) > max_history:
            session.conversation_history = session.conversation_history[-max_history:]
        
        session.last_activity = time.time()
    
    def _create_no_context_answer(self, question: str, start_time: float) -> Answer:
        """Create answer when no relevant context is found"""
        return Answer(
            question=question,
            answer="I couldn't find relevant information in the available documents to answer your question. Please try rephrasing your question or ensure the relevant documents have been uploaded and processed.",
            confidence=AnswerConfidence.VERY_LOW,
            confidence_score=0.0,
            sources=[],
            context_used="",
            processing_time=time.time() - start_time,
            token_usage={},
            follow_up_questions=[],
            reasoning="No relevant context found in document collection"
        )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "conversation_count": len(session.conversation_history),
            "document_filter": session.document_filter
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than specified hours"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_activity > max_age_seconds
        ]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        return len(old_sessions)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from .embedding_manager import EmbeddingManager
    
    async def test_qa_engine():
        # Initialize components
        embedding_manager = EmbeddingManager("test_collection")
        qa_engine = QAEngine(embedding_manager)
        
        # Test question
        question = "What is artificial intelligence?"
        
        # Get answer
        answer = await qa_engine.answer_question(question)
        
        print(f"Question: {answer.question}")
        print(f"Answer: {answer.answer}")
        print(f"Confidence: {answer.confidence.value} ({answer.confidence_score:.3f})")
        print(f"Sources: {len(answer.sources)}")
        print(f"Processing time: {answer.processing_time:.2f}s")
        
        if answer.follow_up_questions:
            print("Follow-up questions:")
            for fq in answer.follow_up_questions:
                print(f"  - {fq}")
    
    # Run test
    # asyncio.run(test_qa_engine())