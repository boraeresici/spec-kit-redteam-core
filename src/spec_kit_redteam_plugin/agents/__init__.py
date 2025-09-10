"""
Multi-agent collaborative specification generation system.
"""

from .base_agent import BaseAgent, AgentResponse, AgentCapability
from .pm_agent import ProductManagerAgent
from .technical_agent import TechnicalAgent
from .security_agent import SecurityAgent
from .qa_agent import QAAgent

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    'pm': ProductManagerAgent,
    'product_manager': ProductManagerAgent,
    'technical': TechnicalAgent,
    'tech': TechnicalAgent,
    'security': SecurityAgent,
    'sec': SecurityAgent,
    'qa': QAAgent,
    'quality_assurance': QAAgent,
}

# Default agent configurations
DEFAULT_AGENT_CONFIGS = {
    'pm': {
        'model': 'gpt-4-turbo',
        'max_tokens': 800,
        'temperature': 0.3,
        'focus_areas': ['user_value', 'business_requirements', 'user_journey', 'acceptance_criteria']
    },
    'technical': {
        'model': 'gpt-4',
        'max_tokens': 1200,
        'temperature': 0.1,
        'focus_areas': ['architecture', 'implementation', 'constraints', 'technical_feasibility']
    },
    'security': {
        'model': 'claude-3-sonnet',
        'max_tokens': 1000,
        'temperature': 0.2,
        'focus_areas': ['threat_modeling', 'compliance', 'data_protection', 'authentication']
    },
    'qa': {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 600,
        'temperature': 0.4,
        'focus_areas': ['test_scenarios', 'edge_cases', 'validation', 'quality_metrics']
    }
}

__all__ = [
    'BaseAgent',
    'AgentResponse', 
    'AgentCapability',
    'ProductManagerAgent',
    'TechnicalAgent',
    'SecurityAgent',
    'QAAgent',
    'AGENT_REGISTRY',
    'DEFAULT_AGENT_CONFIGS'
]