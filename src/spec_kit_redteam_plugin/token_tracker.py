#!/usr/bin/env python3
"""
Token usage tracking and cost management for collaborative AI spec generation.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP

import platformdirs


@dataclass
class TokenUsage:
    """Individual token usage record"""
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    timestamp: datetime
    operation: str  # e.g., 'initial_analysis', 'consensus_round_1'
    
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['cost_usd'] = str(self.cost_usd)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenUsage':
        data['cost_usd'] = Decimal(str(data['cost_usd']))
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CostCalculator:
    """Calculate costs for different AI models"""
    
    # Pricing per 1K tokens (as of 2024)
    MODEL_PRICING = {
        'gpt-4': {'input': Decimal('0.03'), 'output': Decimal('0.06')},
        'gpt-4-turbo': {'input': Decimal('0.01'), 'output': Decimal('0.03')},
        'gpt-3.5-turbo': {'input': Decimal('0.001'), 'output': Decimal('0.002')},
        'claude-3-opus': {'input': Decimal('0.015'), 'output': Decimal('0.075')},
        'claude-3-sonnet': {'input': Decimal('0.003'), 'output': Decimal('0.015')},
        'claude-3-haiku': {'input': Decimal('0.00025'), 'output': Decimal('0.00125')},
    }
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> Decimal:
        """Calculate cost for token usage"""
        if model not in self.MODEL_PRICING:
            # Use GPT-4 pricing as default for unknown models
            model = 'gpt-4'
        
        pricing = self.MODEL_PRICING[model]
        
        # Calculate cost per 1K tokens
        input_cost = (Decimal(str(input_tokens)) / 1000) * pricing['input']
        output_cost = (Decimal(str(output_tokens)) / 1000) * pricing['output']
        
        total_cost = input_cost + output_cost
        return total_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def estimate_cost(self, estimated_tokens: int, model: str, 
                     input_ratio: float = 0.6) -> Decimal:
        """Estimate cost based on total expected tokens"""
        input_tokens = int(estimated_tokens * input_ratio)
        output_tokens = int(estimated_tokens * (1 - input_ratio))
        return self.calculate_cost(input_tokens, output_tokens, model)


class BudgetExceededException(Exception):
    """Raised when session budget is exceeded"""
    def __init__(self, used: Decimal, limit: Decimal):
        self.used = used
        self.limit = limit
        super().__init__(f"Budget exceeded: ${used} used, ${limit} limit")


class TokenTracker:
    """Track token usage and manage budgets for collaborative AI sessions"""
    
    def __init__(self, session_id: Optional[str] = None, session_budget: Optional[float] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_budget = Decimal(str(session_budget)) if session_budget else None
        self.cost_calculator = CostCalculator()
        
        # Storage
        self.usage_records: List[TokenUsage] = []
        self.session_start = datetime.now()
        
        # Cache directory for session data
        self.cache_dir = Path(platformdirs.user_cache_dir("specify-cli")) / "token_sessions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_session_data()
    
    def _load_session_data(self):
        """Load existing session data if available"""
        session_file = self.cache_dir / f"{self.session_id}.json"
        if session_file.exists():
            try:
                with open(session_file) as f:
                    data = json.load(f)
                
                self.usage_records = [
                    TokenUsage.from_dict(record) 
                    for record in data.get('usage_records', [])
                ]
                self.session_start = datetime.fromisoformat(data.get('session_start', self.session_start.isoformat()))
                
            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupt session file, start fresh
                pass
    
    def _save_session_data(self):
        """Save session data to cache"""
        session_file = self.cache_dir / f"{self.session_id}.json"
        
        data = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'session_budget': str(self.session_budget) if self.session_budget else None,
            'usage_records': [record.to_dict() for record in self.usage_records]
        }
        
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def set_session_budget(self, budget: float):
        """Set session budget limit"""
        self.session_budget = Decimal(str(budget))
    
    def track_agent_interaction(self, 
                               agent_name: str,
                               model: str,
                               input_tokens: int,
                               output_tokens: int,
                               operation: str = "generation") -> TokenUsage:
        """Track tokens used in an agent interaction"""
        
        cost = self.cost_calculator.calculate_cost(input_tokens, output_tokens, model)
        
        usage = TokenUsage(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=datetime.now(),
            operation=operation
        )
        
        self.usage_records.append(usage)
        self._save_session_data()
        
        # Check budget after recording
        if self.session_budget and self.get_total_cost() > self.session_budget:
            raise BudgetExceededException(self.get_total_cost(), self.session_budget)
        
        return usage
    
    def check_budget(self, estimated_tokens: int, model: str) -> bool:
        """Check if estimated operation would exceed budget"""
        if not self.session_budget:
            return True
        
        estimated_cost = self.cost_calculator.estimate_cost(estimated_tokens, model)
        projected_total = self.get_total_cost() + estimated_cost
        
        return projected_total <= self.session_budget
    
    def get_remaining_budget(self) -> Decimal:
        """Get remaining budget amount"""
        if not self.session_budget:
            return Decimal('999999')  # Unlimited
        
        return self.session_budget - self.get_total_cost()
    
    def get_total_cost(self) -> Decimal:
        """Get total cost of current session"""
        return sum(record.cost_usd for record in self.usage_records)
    
    def get_total_tokens(self) -> int:
        """Get total tokens used in current session"""
        return sum(record.total_tokens() for record in self.usage_records)
    
    def get_agent_breakdown(self) -> Dict[str, Dict]:
        """Get cost/token breakdown by agent"""
        breakdown = {}
        
        for record in self.usage_records:
            if record.agent_name not in breakdown:
                breakdown[record.agent_name] = {
                    'total_tokens': 0,
                    'total_cost': Decimal('0'),
                    'operations': 0,
                    'model': record.model
                }
            
            agent_data = breakdown[record.agent_name]
            agent_data['total_tokens'] += record.total_tokens()
            agent_data['total_cost'] += record.cost_usd
            agent_data['operations'] += 1
        
        # Convert Decimal to float for JSON serialization
        for agent_data in breakdown.values():
            agent_data['total_cost'] = float(agent_data['total_cost'])
        
        return breakdown
    
    def get_model_breakdown(self) -> Dict[str, Dict]:
        """Get cost/token breakdown by model"""
        breakdown = {}
        
        for record in self.usage_records:
            if record.model not in breakdown:
                breakdown[record.model] = {
                    'total_tokens': 0,
                    'total_cost': Decimal('0'),
                    'usage_count': 0
                }
            
            model_data = breakdown[record.model]
            model_data['total_tokens'] += record.total_tokens()
            model_data['total_cost'] += record.cost_usd
            model_data['usage_count'] += 1
        
        # Convert Decimal to float for JSON serialization
        for model_data in breakdown.values():
            model_data['total_cost'] = float(model_data['total_cost'])
        
        return breakdown
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        duration = datetime.now() - self.session_start
        
        return {
            'session_id': self.session_id,
            'duration_seconds': int(duration.total_seconds()),
            'total_tokens': self.get_total_tokens(),
            'total_cost': float(self.get_total_cost()),
            'budget_limit': float(self.session_budget) if self.session_budget else None,
            'budget_used_percentage': (
                float(self.get_total_cost() / self.session_budget * 100) 
                if self.session_budget else None
            ),
            'agent_breakdown': self.get_agent_breakdown(),
            'model_breakdown': self.get_model_breakdown(),
            'operation_count': len(self.usage_records),
            'session_start': self.session_start.isoformat()
        }
    
    def is_active(self) -> bool:
        """Check if session is still active (used for live displays)"""
        # Session is active if it was created within the last hour and has recent activity
        if not self.usage_records:
            return True  # New session
        
        last_activity = max(record.timestamp for record in self.usage_records)
        return (datetime.now() - last_activity) < timedelta(minutes=30)
    
    def cleanup_old_sessions(self, max_age_days: int = 7):
        """Clean up old session files"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for session_file in self.cache_dir.glob("session_*.json"):
            try:
                if session_file.stat().st_mtime < cutoff.timestamp():
                    session_file.unlink()
            except OSError:
                continue  # Skip files we can't access