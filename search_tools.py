"""
Search tools for the Medical Agents system using OpenAI Agents SDK.

This module provides a unified search tool that can be configured for different
search strategies and behaviors. The main tool includes:
- Document retrieval from Milvus with configurable parameters
- Optional query rewriting for better search results
- Optional document evaluation for relevance
- Query similarity checking to avoid duplicates
- Adaptive search behavior based on configuration
"""
import os
import json
import logging
import random

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from pymilvus import MilvusClient
from agents import function_tool
from agents.models import _openai_shared

from retriever import MedCPTRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    model_name: str = "gpt-4o-mini"
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    allowed_sources: List[str] = None
    retrieve_topk: int = 100
    rerank_topk: int = 25
    query_similarity_threshold: float = 0.85
    similarity_strategy: str = "reuse"  # "reuse", "generate", "none"
    rewrite: bool = True  # Only use rewrite
    review: bool = True  # Only use review
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    relevance_threshold: int = 5
    
    def __post_init__(self):
        if self.allowed_sources is None:
            self.allowed_sources = ["cpg", "statpearls", "recop", "textbooks"]

class SearchTool:
    """Search tool class that maintains its own query cache and configuration."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize the search tool with configuration."""
        self.config = config or SearchConfig()
        self._milvus_client = None
        self._retriever = None
        self._query_cache = []
    
    def _get_milvus_client(self):
        """Get or create Milvus client."""
        if self._milvus_client is None:
            self._milvus_client = MilvusClient(uri=self.config.milvus_uri)
        return self._milvus_client

    def _get_retriever(self):
        """Get or create MedCPT retriever."""
        if self._retriever is None:
            self._retriever = MedCPTRetriever(
                client=self._get_milvus_client(),
                device=self.config.device
            )
        return self._retriever

    def _get_openai_client(self):
        """Get the default OpenAI client."""
        return _openai_shared.get_default_openai_client()

    async def _rewrite_query(self, query: str, domain: Optional[str] = None) -> str:
        """Internal function to rewrite a search query."""
        client = self._get_openai_client()

        domain_instruction = ""
        if domain:
            domain_instruction = f" Focus particularly on aspects related to {domain} and emphasize terminology specific to this field."
        
        prompt = (
            f"Rewrite the following medical search query to improve document retrieval.{domain_instruction}\n\n"
            f"Original query: {query}\n\n"
            f"Instructions:\n"
            f"1. Expand medical terminology and include synonyms\n"
            f"2. Add relevant anatomical, physiological, or pathological terms\n"
            f"3. Include common abbreviations and full forms\n"
            f"4. Maintain the core meaning while making it more searchable\n"
            f"5. Focus on key medical concepts that would appear in medical literature\n\n"
            f"Rewritten query:"
        )
        
        response = await client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a medical query optimization specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()

    async def _evaluate_document_relevance(self, document: str, query: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """Internal function to evaluate document relevance."""
        client = self._get_openai_client()
        
        domain_context = ""
        if domain:
            domain_context = f" Pay special attention to information relevant to {domain}."
        
        prompt = (
            f"Evaluate the relevance of the following document to the given query.{domain_context}\n\n"
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            f"Please assess:\n"
            f"1. How helpful is this document in answering the query?\n"
            f"2. What specific information from the document is relevant?\n"
            f"3. Rate the relevance from 1-10 (10 being most relevant)\n"
            f"4. Provide a brief justification for your rating\n\n"
            f"Respond in JSON format with keys: relevance_score, helpfulness, relevant_info, justification"
        )
        
        response = await client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a medical document evaluation expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            return {
                "relevance_score": 5,
                "helpfulness": "Unable to parse evaluation",
                "relevant_info": "",
                "justification": "Error in evaluation processing"
            }

    def _check_query_similarity(self, query: str) -> Dict[str, Any]:
        """Internal function to check query similarity."""
        if not self._query_cache:
            return {
                "is_similar": False,
                "similarity_score": 0.0,
                "most_similar_query": None,
                "recommendation": "proceed"
            }
        
        retriever = self._get_retriever()
        current_embedding = retriever.encode(query)
        
        similarities = []
        for cached_item in self._query_cache:
            prev_embedding = cached_item["embedding"]
            similarity = torch.cosine_similarity(
                current_embedding.unsqueeze(0), 
                prev_embedding.unsqueeze(0)
            ).item()
            similarities.append((similarity, cached_item["query"]))
        
        max_similarity, most_similar_query = max(similarities, key=lambda x: x[0])
        
        is_similar = max_similarity > self.config.query_similarity_threshold
        
        recommendation = "proceed"
        if is_similar:
            if self.config.similarity_strategy == "reuse":
                recommendation = "reuse_previous_results"
            elif self.config.similarity_strategy == "generate":
                recommendation = "generate_new_query"
        
        return {
            "is_similar": is_similar,
            "similarity_score": max_similarity,
            "most_similar_query": most_similar_query,
            "recommendation": recommendation
        }

    async def _generate_distinct_query(self, original_query: str, previous_queries: List[str], domain: Optional[str] = None) -> str:
        """Internal function to generate a distinct query."""
        client = self._get_openai_client()
        
        domain_context = ""
        if domain:
            domain_context = f" Focus on aspects related to {domain}."
        
        prompt = (
            f"Your previous question '{original_query}' is too similar to questions already asked.{domain_context}\n\n"
            f"Previous queries include: {', '.join(previous_queries)}\n\n"
            f"Please generate a completely different question that explores a new aspect of the main problem.\n"
            f"Ensure your new query addresses a unique angle not covered by these queries.\n\n"
            f"New query:"
        )
        
        response = await client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a medical query generation specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()

    def get_previous_queries(self, max_length: int = 1000, max_queries: Optional[int] = None) -> str:
        """
        Get all previous queries and their retrieved documents in an organized string format.
        
        Returns:
            Formatted string containing all previous queries and their results
        """
        if not self._query_cache:
            return "No previous queries found."
        
        result = "Previous Search Queries and Results:\n"
        result += "=" * 50 + "\n\n"
        
        for i, cached_item in enumerate(random.sample(self._query_cache, min(max_queries, len(self._query_cache))) if max_queries else self._query_cache, 1):
            result += f"Query {i}: {cached_item['query']}\n"
            result += "-" * 30 + "\n"
            
            documents = cached_item.get('documents', [])
            if documents:
                result += f"Retrieved {len(documents)} documents:\n\n"
                for j, doc in enumerate(documents, 1):
                    doc_preview = doc[:max_length] + "..." if len(doc) > max_length else doc
                    result += f"  Document {j}:\n"
                    result += f"  {doc_preview}\n\n"
            else:
                result += "No documents retrieved for this query.\n\n"
            
            result += "=" * 50 + "\n\n"
        
        return result

    async def search_medical_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> List[str]:
        """
        Search for relevant medical knowledge with configurable behavior.
        
        This is the main search method that can be configured for different search strategies:
        - Basic retrieval with optional query rewriting
        - Document relevance review and filtering
        - Query similarity checking to avoid duplicates
        - Configurable sources and result limits
        
        Args:
            query: The search query
            domain: Optional medical domain to focus the search
        
        Returns:
            List of relevant document texts
        """
        
        
        if self.config.similarity_strategy != "none":
            similarity_info = self._check_query_similarity(query)

            if similarity_info["is_similar"] and similarity_info["recommendation"] == "reuse_previous_results":
                for cached_item in self._query_cache:
                    if cached_item["query"] == similarity_info["most_similar_query"]:
                        logger.info(f"Reusing results from similar query: {similarity_info['most_similar_query']}")
                        return [doc for doc in cached_item["documents"]]

            if similarity_info["is_similar"] and similarity_info["recommendation"] == "generate_new_query":
                previous_queries = [item["query"] for item in self._query_cache]
                query = await self._generate_distinct_query(query, previous_queries, domain)
                logger.info(f"Generated new query: {query}")

        original_query = query

        if domain:
            query += f" Specifically considering aspects related to {domain}."

        if self.config.rewrite:
            query = await self._rewrite_query(query, domain)
            logger.info(f"Rewritten query: {query}")

        retriever = self._get_retriever()

        unique_docs = retriever.retrieve(
            query, sources=self.config.allowed_sources, topk=self.config.retrieve_topk
        )
        reranked_docs = retriever.rerank(query, unique_docs)
        documents = reranked_docs[:self.config.rerank_topk]

        if self.config.review:
            reviewed_docs = []
            for doc in documents:
                evaluation = await self._evaluate_document_relevance(doc, original_query, domain)
                if evaluation.get("relevance_score", 0) >= self.config.relevance_threshold:
                    reviewed_docs.append(doc)
            documents = reviewed_docs
            logger.info(f"After review: {len(documents)}/{len(reranked_docs)} documents deemed helpful")

        self._query_cache.append({
            "query": original_query,
            "embedding": retriever.encode(original_query),
            "documents": documents
        })
        return documents

    def get_search_function(self):
        """Get the search function as a function_tool for use with OpenAI Agents SDK."""
        @function_tool
        async def search_medical_knowledge(
            query: str,
            domain: Optional[str] = None,
        ) -> List[str]:
            """
            Search for relevant medical knowledge with configurable behavior.
            
            This is the main search tool that can be configured for different search strategies:
            - Basic retrieval with optional query rewriting
            - Document relevance review and filtering
            - Query similarity checking to avoid duplicates
            - Configurable sources and result limits
            
            Args:
                query: The search query
                domain: Optional medical domain to focus the search
            
            Returns:
                List of relevant document texts
            """
            return await self.search_medical_knowledge(
                query=query,
                domain=domain,
            )
        
        return search_medical_knowledge

def get_search_tools(search_tool: Optional[SearchTool] = None) -> List:
    """
    Get the search tools for use with OpenAI Agents SDK.
    
    Returns:
        List containing the main search tool and previous queries tool
    """
    return [search_tool.get_search_function()] if search_tool else []

def create_search_tool_instance(config: Optional[SearchConfig] = None) -> SearchTool:
    """
    Create a new SearchTool instance with its own query cache.
    
    Args:
        config: Optional configuration for the search tool
    
    Returns:
        SearchTool instance
    """
    return SearchTool(config)
