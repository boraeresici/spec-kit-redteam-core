#!/usr/bin/env python3
"""
Base agent class for collaborative AI specification generation.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..token_tracker import TokenTracker, BudgetExceededException


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_format: str
    output_format: str
    estimated_tokens: int


@dataclass
class AgentResponse:
    """Response from an AI agent"""
    agent_name: str
    operation: str
    content: str
    metadata: Dict[str, Any]
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'agent_name': self.agent_name,
            'operation': self.operation,
            'content': self.content,
            'metadata': self.metadata,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'model': self.model,
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentResponse':
        return cls(**data)


class BaseAgent(ABC):
    """Base class for all collaborative AI agents"""
    
    def __init__(self, 
                 role: str,
                 model: str,
                 token_tracker: TokenTracker,
                 max_tokens: int = 1000,
                 temperature: float = 0.3):
        self.role = role
        self.model = model
        self.token_tracker = token_tracker
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Agent state
        self.conversation_history: List[Dict] = []
        self.capabilities: List[AgentCapability] = []
        self.focus_areas: List[str] = []
        
        # Load agent-specific configuration
        self._load_configuration()
    
    @abstractmethod
    def _load_configuration(self):
        """Load agent-specific configuration and capabilities"""
        pass
    
    @abstractmethod
    def analyze_input(self, user_input: str, context: Dict = None) -> AgentResponse:
        """Analyze user input from agent's perspective"""
        pass
    
    @abstractmethod
    def review_other_agent_response(self, 
                                   other_response: AgentResponse,
                                   original_input: str) -> AgentResponse:
        """Review and comment on another agent's response"""
        pass
    
    @abstractmethod
    def build_consensus_contribution(self, 
                                   all_responses: List[AgentResponse],
                                   original_input: str) -> AgentResponse:
        """Contribute to building consensus from all agent responses"""
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for caching purposes"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _call_ai_api(self, prompt: str, context: Dict = None) -> AgentResponse:
        """
        Mock AI API call - In real implementation, this would call
        OpenAI, Anthropic, or other AI APIs
        """
        
        # For now, create a mock response
        # In real implementation, this would make actual API calls
        input_tokens = self._estimate_tokens(prompt)
        output_tokens = self._estimate_tokens("Mock response content")
        
        # Track token usage
        usage = self.token_tracker.track_agent_interaction(
            agent_name=self.role,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=context.get('operation', 'generation') if context else 'generation'
        )
        
        # Create mock response
        response = AgentResponse(
            agent_name=self.role,
            operation=context.get('operation', 'analysis') if context else 'analysis',
            content=f"Mock {self.role} agent response to: {prompt[:100]}...",
            metadata={
                'model': self.model,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'focus_areas': self.focus_areas
            },
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            timestamp=usage.timestamp.isoformat()
        )
        
        # Store in conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': prompt,
            'timestamp': usage.timestamp.isoformat()
        })
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response.content,
            'timestamp': usage.timestamp.isoformat()
        })
        
        return response
    
    def check_budget_for_operation(self, estimated_input: str) -> bool:
        """Check if operation is within budget"""
        estimated_tokens = self._estimate_tokens(estimated_input) + self.max_tokens
        return self.token_tracker.check_budget(estimated_tokens, self.model)
    
    def get_agent_status(self) -> Dict:
        """Get current agent status and statistics"""
        agent_breakdown = self.token_tracker.get_agent_breakdown()
        my_stats = agent_breakdown.get(self.role, {
            'total_tokens': 0,
            'total_cost': 0,
            'operations': 0
        })
        
        return {
            'role': self.role,
            'model': self.model,
            'tokens_used': my_stats['total_tokens'],
            'cost_incurred': my_stats['total_cost'],
            'operations_completed': my_stats['operations'],
            'conversation_turns': len(self.conversation_history) // 2,
            'capabilities': [cap.name for cap in self.capabilities],
            'focus_areas': self.focus_areas
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def export_conversation(self) -> Dict:
        """Export conversation history for analysis"""
        return {
            'agent_role': self.role,
            'model': self.model,
            'conversation_history': self.conversation_history,
            'final_status': self.get_agent_status()
        }


class AgentResponseValidator:
    """Validates agent responses for quality and completeness"""
    
    @staticmethod
    def validate_response(response: AgentResponse) -> Dict[str, Any]:
        """Validate an agent response"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0
        }
        
        # Check required fields
        required_fields = ['agent_name', 'content', 'operation']
        for field in required_fields:
            if not getattr(response, field, None):
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Check content length
        if len(response.content) < 50:
            validation_result['warnings'].append("Response content seems too short")
            validation_result['quality_score'] -= 0.1
        
        if len(response.content) > 5000:
            validation_result['warnings'].append("Response content is very long")
            validation_result['quality_score'] -= 0.1
        
        # Check token usage reasonableness
        if response.input_tokens + response.output_tokens > 10000:
            validation_result['warnings'].append("High token usage detected")
        
        # Check confidence level
        if response.confidence < 0.7:
            validation_result['warnings'].append("Low confidence response")
            validation_result['quality_score'] -= 0.2
        
        validation_result['quality_score'] = max(0.0, validation_result['quality_score'])
        
        return validation_result
    
    @staticmethod
    def validate_consensus_readiness(responses: List[AgentResponse]) -> Dict[str, Any]:
        """Check if responses are ready for consensus building"""
        validation_result = {
            'ready_for_consensus': True,
            'issues': [],
            'agent_coverage': {}
        }
        
        # Check agent diversity
        agent_roles = {r.agent_name for r in responses}
        required_roles = {'ProductManager', 'Technical'}  # Minimum for MVP
        
        missing_roles = required_roles - agent_roles
        if missing_roles:
            validation_result['ready_for_consensus'] = False
            validation_result['issues'].append(f"Missing required agent roles: {missing_roles}")
        
        # Check response quality
        low_quality_responses = []
        for response in responses:
            quality_check = AgentResponseValidator.validate_response(response)
            if quality_check['quality_score'] < 0.6:
                low_quality_responses.append(response.agent_name)
        
        if low_quality_responses:
            validation_result['issues'].append(f"Low quality responses from: {low_quality_responses}")
        
        # Agent coverage analysis
        for response in responses:
            validation_result['agent_coverage'][response.agent_name] = {
                'content_length': len(response.content),
                'confidence': response.confidence,
                'token_efficiency': response.output_tokens / max(1, response.input_tokens)
            }
        
        return validation_result