from typing import List, Dict, Optional
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from langdetect import detect


class RAGEngine:
    
    def __init__(self, config, vector_store):
        self.config = config
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = OpenAI(
            api_key=config.openai_api_key,
            model=config.openai_model,
            temperature=0.1
        )
        
        # QA prompt
        self.qa_prompt = PromptTemplate(
            """You are a helpful assistant that answers questions based ONLY on the provided context.
You are capable of understanding and responding in both Arabic and English.
            
Context from documents:
{context}

Conversation history:
{history}

Language Note: {language_note}

Current question: {question}

Instructions:
1. Answer ONLY based on the information in the context provided
2. Be precise and cite specific information from the context
3. Consider the conversation history for follow-up questions
4. Do not make up or infer information not explicitly stated in the context
5. Respond in the same language as the question when possible
6. If the context contains mixed languages, prioritize clarity in your response

Answer:"""
        )
    
    def _detect_query_language(self, query: str) -> str:
        try:
            # Check for Arabic characters
            has_arabic = any('\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' for char in query)
            if has_arabic:
                return 'ar'
            
            # Use langdetect for other cases
            detected = detect(query)
            return detected
        except:
            return 'en'  # Default to English
    
    def _organize_by_language(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        organized = {
            'ar': [],
            'en': [],
            'mixed': [],
            'other': []
        }
        
        for doc in documents:
            lang = doc.get('language', 'unknown')
            if lang == 'ar':
                organized['ar'].append(doc)
            elif lang == 'en':
                organized['en'].append(doc)
            elif lang == 'mixed':
                organized['mixed'].append(doc)
            else:
                organized['other'].append(doc)
        
        return organized
    
    def _build_language_aware_context(self, documents: List[Dict], query_language: str) -> tuple[str, str]:
        organized_docs = self._organize_by_language(documents)
        
        # Determine context building strategy
        context_parts = []
        language_note = ""
        
        # Count documents by language
        ar_count = len(organized_docs['ar'])
        en_count = len(organized_docs['en'])
        mixed_count = len(organized_docs['mixed'])
        
        # Prioritize documents matching query language
        if query_language == 'ar':
            # Arabic query - prioritize Arabic docs
            primary_docs = organized_docs['ar'][:self.config.top_k_retrieval]
            secondary_docs = organized_docs['mixed'][:2]  # Include some mixed docs
            
            if not primary_docs and secondary_docs:
                language_note = "The context contains mixed Arabic/English content."
            elif not primary_docs:
                # Fallback to English if no Arabic docs
                primary_docs = organized_docs['en'][:self.config.top_k_retrieval]
                language_note = "No Arabic content found. Using English documents."
        else:
            # English query - prioritize English docs
            primary_docs = organized_docs['en'][:self.config.top_k_retrieval]
            secondary_docs = organized_docs['mixed'][:2]  # Include some mixed docs
            
            if not primary_docs and secondary_docs:
                language_note = "The context contains mixed Arabic/English content."
            elif not primary_docs:
                # Fallback to Arabic if no English docs
                primary_docs = organized_docs['ar'][:self.config.top_k_retrieval]
                language_note = "No English content found. Using Arabic documents."
        
        # Build context from selected documents
        all_selected_docs = primary_docs + secondary_docs
        
        # Sort by score to maintain relevance
        all_selected_docs.sort(key=lambda x: x.get('final_score', x['score']), reverse=True)
        
        # Take top K after language-aware selection
        final_docs = all_selected_docs[:self.config.top_k_retrieval]
        
        # Build context
        for i, doc in enumerate(final_docs):
            lang_marker = f"[{doc.get('language', 'unknown').upper()}]" if doc.get('is_mixed', False) else ""
            context_parts.append(f"[{i+1}] {lang_marker} {doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Add language distribution info if mixed
        if mixed_count > 0 or (ar_count > 0 and en_count > 0):
            if not language_note:
                language_note = f"Context contains documents in: Arabic ({ar_count}), English ({en_count}), Mixed ({mixed_count})"
        
        return context, language_note
    
    async def query(self, question: str, conversation_history: List[Dict] = None) -> Dict:
        try:
            # Detect query language
            query_language = self._detect_query_language(question)
            
            # Search for relevant documents
            docs = await self.vector_store.search(
                question, 
                top_k=self.config.top_k_retrieval
            )
            
            # Remove duplicates and rank by score
            unique_docs = self._deduplicate_documents(docs)
            
            # Check if we found relevant documents
            if not unique_docs:
                return {
                    "answer": "No answer found in the uploaded documents.",
                    "citations": [],
                    "confidence": 0.0
                }
            
            # Calculate average relevance score
            avg_score = sum(doc.get("final_score", doc["score"]) for doc in unique_docs) / len(unique_docs)
            
            print(f"Query language: {query_language}, Average score: {avg_score}")
            
            # Check confidence threshold
            if avg_score < self.config.confidence_threshold:
                return {
                    "answer": "No answer found in the uploaded documents.",
                    "citations": [],
                    "confidence": avg_score
                }
            
            # Build language-aware context
            context, language_note = self._build_language_aware_context(unique_docs, query_language)
            
            # Extract citations from the documents used in context
            citations = []
            seen_citations = set()
            
            for doc in unique_docs[:self.config.top_k_retrieval]:
                citation_key = f"{doc['filename']}_{doc['page']}"
                if citation_key not in seen_citations:
                    citations.append({
                        "filename": doc["filename"],
                        "page": doc["page"]
                    })
                    seen_citations.add(citation_key)
            
            # Format conversation history
            history_text = ""
            if conversation_history:
                history_parts = []
                for msg in conversation_history[-4:]:  # Last 2 exchanges
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    # Clean any citations from history
                    content = msg["content"].split("\n\n**Citations:**")[0]
                    history_parts.append(f"{role}: {content}")
                history_text = "\n".join(history_parts)
            
            # Generate answer using LLM
            response = self.llm.complete(
                self.qa_prompt.format(
                    context=context,
                    language_note=language_note,
                    history=history_text,
                    question=question
                )
            )
            
            answer = response.text.strip()
            
            # Double-check for hallucination
            if answer and answer != "No answer found in the uploaded documents.":
                # Verify the answer is grounded in context
                if not self._verify_grounding(answer, context):
                    return {
                        "answer": "No answer found in the uploaded documents.",
                        "citations": [],
                        "confidence": 0.5
                    }
            
            return {
                "answer": answer,
                "citations": citations,
                "confidence": avg_score
            }
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }
    
    def _deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        seen = set()
        unique_docs = []
        
        for doc in documents:
            key = f"{doc['filename']}_{doc['page']}_{doc['chunk_index']}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        # Sort by final_score if available, otherwise by score
        return sorted(unique_docs, 
                     key=lambda x: x.get("final_score", x["score"]), 
                     reverse=True)
    
    def _verify_grounding(self, answer: str, context: str) -> bool:
        # Basic check ensure key phrases from answer appear in context
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Skip verification for very short answers
        if len(answer.split()) < 5:
            return True
        
        # For Arabic text, we need different handling
        has_arabic_answer = any('\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' for char in answer)
        has_arabic_context = any('\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' for char in context)
        
        # If languages don't match, be more lenient (translation might be happening)
        if has_arabic_answer != has_arabic_context:
            return True
        
        # Check if significant words from answer appear in context
        significant_words = [
            word for word in answer_lower.split() 
            if len(word) > 4 and word not in ['that', 'this', 'these', 'those', 'which', 'where', 'when', 'الذي', 'التي', 'هذا', 'هذه', 'ذلك', 'تلك']
        ]
        
        if not significant_words:
            return True
        
        # At least 40% of significant words should be in context (reduced threshold for multilingual)
        found = sum(1 for word in significant_words if word in context_lower)
        return found >= len(significant_words) * 0.4