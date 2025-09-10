#!/usr/bin/env python3
"""
Multi-Agent Collaborative Specification Orchestrator.
Coordinates multiple AI agents to generate comprehensive specifications.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .token_tracker import TokenTracker, BudgetExceededException
from .agents.base_agent import BaseAgent, AgentResponse, AgentResponseValidator
from .agents import AGENT_REGISTRY, DEFAULT_AGENT_CONFIGS
from .caching import SpecCache


@dataclass
class ConflictItem:
    """Represents a conflict between agent responses"""
    agent_1: str
    agent_2: str
    conflict_type: str
    description: str
    resolution_strategy: str
    severity: str  # 'low', 'medium', 'high'


@dataclass
class ConsensusResult:
    """Result of consensus building process"""
    final_agreement: str
    confidence_score: float
    rounds_completed: int
    unresolved_conflicts: List[ConflictItem]
    agent_contributions: Dict[str, float]  # contribution weight per agent
    
    
@dataclass
class SpecGenerationResult:
    """Final result of collaborative spec generation"""
    specification: str
    consensus_result: ConsensusResult
    token_summary: Dict[str, Any]
    discussion_log: List[Dict]
    generation_metadata: Dict[str, Any]
    quality_score: float


class ConflictDetector:
    """Detects and analyzes conflicts between agent responses"""
    
    def __init__(self):
        self.conflict_patterns = {
            'priority_mismatch': ['priority', 'important', 'critical', 'optional'],
            'approach_difference': ['should', 'must', 'could', 'recommend', 'suggest'],
            'technical_disagreement': ['architecture', 'framework', 'technology', 'approach'],
            'scope_disagreement': ['scope', 'include', 'exclude', 'boundary'],
            'timeline_conflict': ['time', 'phase', 'schedule', 'deadline', 'milestone']
        }
    
    def detect_conflicts(self, responses: List[AgentResponse]) -> List[ConflictItem]:
        """Detect conflicts between agent responses"""
        conflicts = []
        
        for i, response_1 in enumerate(responses):
            for j, response_2 in enumerate(responses[i+1:], i+1):
                conflict_items = self._analyze_response_pair(response_1, response_2)
                conflicts.extend(conflict_items)
        
        return conflicts
    
    def _analyze_response_pair(self, response_1: AgentResponse, response_2: AgentResponse) -> List[ConflictItem]:
        """Analyze a pair of responses for conflicts"""
        conflicts = []
        
        content_1_lower = response_1.content.lower()
        content_2_lower = response_2.content.lower()
        
        for conflict_type, keywords in self.conflict_patterns.items():
            # Simple conflict detection based on opposing keywords
            conflicts_detected = self._detect_keyword_conflicts(
                content_1_lower, content_2_lower, keywords, conflict_type
            )
            
            for conflict_desc in conflicts_detected:
                conflicts.append(ConflictItem(
                    agent_1=response_1.agent_name,
                    agent_2=response_2.agent_name,
                    conflict_type=conflict_type,
                    description=conflict_desc,
                    resolution_strategy=self._suggest_resolution_strategy(conflict_type),
                    severity=self._assess_severity(conflict_type, conflict_desc)
                ))
        
        return conflicts
    
    def _detect_keyword_conflicts(self, content_1: str, content_2: str, 
                                 keywords: List[str], conflict_type: str) -> List[str]:
        """Detect conflicts based on keyword analysis"""
        conflicts = []
        
        # Simple heuristic: look for contradictory statements
        for keyword in keywords:
            if keyword in content_1 and keyword in content_2:
                # More sophisticated analysis would be needed here
                # For now, flag potential conflicts for review
                conflicts.append(f"Both agents mention '{keyword}' - potential {conflict_type}")
        
        return conflicts
    
    def _suggest_resolution_strategy(self, conflict_type: str) -> str:
        """Suggest resolution strategy for conflict type"""
        strategies = {
            'priority_mismatch': 'Facilitate priority alignment discussion',
            'approach_difference': 'Evaluate approaches based on requirements',
            'technical_disagreement': 'Technical feasibility assessment',
            'scope_disagreement': 'Clarify scope boundaries with stakeholders',
            'timeline_conflict': 'Review timeline constraints and dependencies'
        }
        return strategies.get(conflict_type, 'General consensus building')
    
    def _assess_severity(self, conflict_type: str, description: str) -> str:
        """Assess severity of conflict"""
        high_severity_types = ['technical_disagreement', 'scope_disagreement']
        if conflict_type in high_severity_types:
            return 'high'
        return 'medium'


class ConsensusBuilder:
    """Builds consensus from multiple agent responses"""
    
    def __init__(self, token_tracker: TokenTracker):
        self.token_tracker = token_tracker
        self.conflict_detector = ConflictDetector()
    
    async def build_consensus(self, 
                            agents: List[BaseAgent],
                            initial_responses: List[AgentResponse],
                            original_input: str,
                            max_rounds: int = 3) -> ConsensusResult:
        """Build consensus through iterative discussion"""
        
        current_responses = initial_responses.copy()
        rounds_completed = 0
        discussion_history = []
        
        # Detect initial conflicts
        conflicts = self.conflict_detector.detect_conflicts(current_responses)
        
        # Iterative consensus building
        for round_num in range(max_rounds):
            if not conflicts:
                break  # Consensus achieved
            
            rounds_completed += 1
            
            # Check budget before continuing
            remaining_budget = self.token_tracker.get_remaining_budget()
            if remaining_budget < 5.0:  # Reserve minimum budget
                break
            
            # Generate consensus contributions from all agents
            consensus_responses = []
            for agent in agents:
                try:
                    consensus_response = agent.build_consensus_contribution(
                        current_responses, original_input
                    )
                    consensus_responses.append(consensus_response)
                    
                except BudgetExceededException:
                    break  # Stop if budget exceeded
            
            # Update responses and re-evaluate conflicts
            current_responses = consensus_responses
            conflicts = self.conflict_detector.detect_conflicts(current_responses)
            
            discussion_history.append({
                'round': round_num + 1,
                'responses': [r.to_dict() for r in consensus_responses],
                'conflicts_detected': len(conflicts)
            })
        
        # Generate final consensus
        final_agreement = self._synthesize_final_agreement(current_responses)
        confidence_score = self._calculate_confidence_score(current_responses, conflicts)
        agent_contributions = self._calculate_agent_contributions(current_responses)
        
        return ConsensusResult(
            final_agreement=final_agreement,
            confidence_score=confidence_score,
            rounds_completed=rounds_completed,
            unresolved_conflicts=conflicts,
            agent_contributions=agent_contributions
        )
    
    def _synthesize_final_agreement(self, responses: List[AgentResponse]) -> str:
        """Synthesize final agreement from agent responses"""
        # Simple synthesis - in production this would be more sophisticated
        synthesis_sections = []
        
        synthesis_sections.append("# Collaborative Specification Consensus")
        synthesis_sections.append(f"Generated by {len(responses)} AI agents")
        synthesis_sections.append("")
        
        for response in responses:
            synthesis_sections.append(f"## {response.agent_name} Perspective")
            # Extract key points from response content
            content_lines = response.content.split('\n')[:10]  # First 10 lines
            synthesis_sections.extend(content_lines)
            synthesis_sections.append("")
        
        return '\n'.join(synthesis_sections)
    
    def _calculate_confidence_score(self, responses: List[AgentResponse], 
                                   conflicts: List[ConflictItem]) -> float:
        """Calculate overall confidence in consensus"""
        base_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Reduce confidence based on unresolved conflicts
        conflict_penalty = len(conflicts) * 0.1
        high_severity_penalty = sum(0.2 for c in conflicts if c.severity == 'high')
        
        final_confidence = max(0.0, base_confidence - conflict_penalty - high_severity_penalty)
        return min(1.0, final_confidence)
    
    def _calculate_agent_contributions(self, responses: List[AgentResponse]) -> Dict[str, float]:
        """Calculate relative contribution weight of each agent"""
        contributions = {}
        total_weight = 0
        
        for response in responses:
            # Weight based on content length, confidence, and token efficiency
            content_weight = min(1.0, len(response.content) / 1000)
            confidence_weight = response.confidence
            efficiency_weight = response.output_tokens / max(1, response.input_tokens)
            
            agent_weight = (content_weight + confidence_weight + efficiency_weight) / 3
            contributions[response.agent_name] = agent_weight
            total_weight += agent_weight
        
        # Normalize to sum to 1.0
        if total_weight > 0:
            contributions = {name: weight/total_weight for name, weight in contributions.items()}
        
        return contributions


class CollaborativeSpecOrchestrator:
    """Main orchestrator for collaborative specification generation"""
    
    def __init__(self, 
                 token_tracker: TokenTracker,
                 max_concurrent_agents: int = 4,
                 enable_caching: bool = True,
                 enable_parallel: bool = True):
        self.token_tracker = token_tracker
        self.max_concurrent_agents = max_concurrent_agents
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        
        self.consensus_builder = ConsensusBuilder(token_tracker)
        self.response_validator = AgentResponseValidator()
        
        # Initialize caching system
        self.cache = SpecCache() if enable_caching else None
        
        # Session state
        self.discussion_log: List[Dict] = []
        self.current_agents: List[BaseAgent] = []
    
    async def generate_collaborative_spec(self, 
                                        user_input: str,
                                        agent_names: List[str] = None,
                                        budget_limit: float = 50.0,
                                        custom_config: Dict = None) -> SpecGenerationResult:
        """Generate specification using collaborative AI agents"""
        
        # Set session budget
        self.token_tracker.set_session_budget(budget_limit)
        
        generation_start = time.time()
        
        try:
            # Phase 1: Initialize agents
            agents = self._initialize_agents(agent_names or ['pm', 'technical'], custom_config)
            
            # Phase 2: Gather initial responses
            initial_responses = await self._gather_initial_responses(agents, user_input)
            
            # Phase 3: Validate responses
            validation_results = self._validate_responses(initial_responses)
            
            # Phase 4: Build consensus
            if validation_results['ready_for_consensus']:
                consensus_result = await self.consensus_builder.build_consensus(
                    agents, initial_responses, user_input
                )
            else:
                # Handle validation failures
                consensus_result = self._create_fallback_consensus(initial_responses)
            
            # Phase 5: Generate final specification
            final_spec = self._format_final_specification(
                consensus_result, initial_responses, user_input
            )
            
            # Calculate quality score
            quality_score = self._calculate_overall_quality_score(
                initial_responses, consensus_result, validation_results
            )
            
            generation_time = time.time() - generation_start
            
            # Gather cache statistics
            cache_stats = self.cache.get_cache_stats() if self.cache else {}
            
            return SpecGenerationResult(
                specification=final_spec,
                consensus_result=consensus_result,
                token_summary=self.token_tracker.get_session_summary(),
                discussion_log=self.discussion_log,
                generation_metadata={
                    'agents_used': [agent.role for agent in agents],
                    'generation_time_seconds': generation_time,
                    'validation_results': validation_results,
                    'budget_utilization': float(self.token_tracker.get_total_cost() / self.token_tracker.session_budget * 100) if self.token_tracker.session_budget else 0,
                    'cache_performance': cache_stats,
                    'optimizations_used': {
                        'caching_enabled': self.enable_caching,
                        'parallel_execution': self.enable_parallel
                    }
                },
                quality_score=quality_score
            )
            
        except BudgetExceededException as e:
            # Handle budget exceeded gracefully
            return self._create_budget_exceeded_result(user_input, str(e))
    
    def _initialize_agents(self, agent_names: List[str], custom_config: Dict = None) -> List[BaseAgent]:
        """Initialize specified agents with configuration"""
        agents = []
        
        for agent_name in agent_names:
            if agent_name not in AGENT_REGISTRY:
                continue
            
            agent_class = AGENT_REGISTRY[agent_name]
            
            # Use custom config or defaults
            config = custom_config.get(agent_name, {}) if custom_config else {}
            default_config = DEFAULT_AGENT_CONFIGS.get(agent_name, {})
            final_config = {**default_config, **config}
            
            # Create agent with token tracker
            agent = agent_class(
                role=agent_name.title(),
                model=final_config.get('model', 'gpt-4'),
                token_tracker=self.token_tracker,
                max_tokens=final_config.get('max_tokens', 1000),
                temperature=final_config.get('temperature', 0.3)
            )
            
            agents.append(agent)
            self.discussion_log.append({
                'event': 'agent_initialized',
                'agent': agent.role,
                'config': final_config,
                'timestamp': datetime.now().isoformat()
            })
        
        self.current_agents = agents
        return agents
    
    async def _gather_initial_responses(self, agents: List[BaseAgent], user_input: str) -> List[AgentResponse]:
        """Gather initial responses from all agents with caching optimization"""
        
        self.discussion_log.append({
            'event': 'initial_analysis_start',
            'agent_count': len(agents),
            'user_input': user_input[:200],  # First 200 chars
            'timestamp': datetime.now().isoformat()
        })
        
        valid_responses = []
        cache_hits = 0
        cache_misses = 0
        
        if self.enable_parallel:
            # Parallel execution with caching
            tasks = []
            for agent in agents:
                task = asyncio.create_task(self._get_agent_response_cached(agent, user_input))
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and count cache statistics
            for response in responses:
                if isinstance(response, tuple):  # (response, was_cached)
                    agent_response, was_cached = response
                    valid_responses.append(agent_response)
                    if was_cached:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                elif isinstance(response, AgentResponse):
                    valid_responses.append(response)
                    cache_misses += 1
                # Skip exceptions
            
        else:
            # Sequential execution with caching
            for agent in agents:
                try:
                    response, was_cached = await self._get_agent_response_cached(agent, user_input)
                    valid_responses.append(response)
                    if was_cached:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                except BudgetExceededException:
                    break  # Stop if budget exceeded
                except Exception:
                    continue  # Skip failed agents
        
        # Log cache performance
        cache_performance = {
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / max(cache_hits + cache_misses, 1) * 100
        }
        
        self.discussion_log.append({
            'event': 'initial_analysis_complete',
            'responses_collected': len(valid_responses),
            'cache_performance': cache_performance,
            'timestamp': datetime.now().isoformat()
        })
        
        return valid_responses
    
    async def _get_agent_response(self, agent: BaseAgent, user_input: str) -> AgentResponse:
        """Get response from single agent (async wrapper)"""
        # Since our base implementation is synchronous, wrap it
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.analyze_input, user_input)
    
    async def _get_agent_response_cached(self, agent: BaseAgent, user_input: str) -> Tuple[AgentResponse, bool]:
        """Get response from agent with caching support"""
        
        # Check cache first if enabled
        if self.cache:
            cached_response = self.cache.get_cached_response(
                agent_role=agent.role,
                input_text=user_input,
                context={'operation': 'initial_analysis'}
            )
            
            if cached_response:
                # Update cached response timestamp to current time
                cached_response.timestamp = datetime.now().isoformat()
                return cached_response, True  # Cache hit
        
        # Cache miss - get fresh response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.analyze_input, user_input)
        
        # Cache the response if caching is enabled
        if self.cache:
            self.cache.cache_response(
                agent_role=agent.role,
                input_text=user_input,
                response=response,
                context={'operation': 'initial_analysis'}
            )
        
        return response, False  # Cache miss
    
    def _validate_responses(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Validate agent responses for quality and completeness"""
        return self.response_validator.validate_consensus_readiness(responses)
    
    def _create_fallback_consensus(self, responses: List[AgentResponse]) -> ConsensusResult:
        """Create fallback consensus when validation fails"""
        return ConsensusResult(
            final_agreement="Validation issues prevented full consensus",
            confidence_score=0.5,
            rounds_completed=0,
            unresolved_conflicts=[],
            agent_contributions={r.agent_name: 1.0/len(responses) for r in responses}
        )
    
    def _format_final_specification(self, 
                                   consensus_result: ConsensusResult,
                                   initial_responses: List[AgentResponse],
                                   user_input: str) -> str:
        """Format final specification document"""
        
        spec_sections = []
        
        # Header
        spec_sections.append("# Collaborative AI Specification")
        spec_sections.append(f"**Generated**: {datetime.now().isoformat()}")
        spec_sections.append(f"**Agents**: {', '.join([r.agent_name for r in initial_responses])}")
        spec_sections.append(f"**Confidence**: {consensus_result.confidence_score:.1%}")
        spec_sections.append("")
        
        # Original request
        spec_sections.append("## Original Request")
        spec_sections.append(user_input)
        spec_sections.append("")
        
        # Consensus result
        spec_sections.append("## Consensus Specification")
        spec_sections.append(consensus_result.final_agreement)
        spec_sections.append("")
        
        # Individual perspectives
        spec_sections.append("## Agent Perspectives")
        for response in initial_responses:
            spec_sections.append(f"### {response.agent_name} Analysis")
            spec_sections.append(response.content[:500] + "..." if len(response.content) > 500 else response.content)
            spec_sections.append("")
        
        # Unresolved issues
        if consensus_result.unresolved_conflicts:
            spec_sections.append("## Outstanding Issues")
            for conflict in consensus_result.unresolved_conflicts:
                spec_sections.append(f"- **{conflict.conflict_type}**: {conflict.description}")
            spec_sections.append("")
        
        return '\n'.join(spec_sections)
    
    def _calculate_overall_quality_score(self, 
                                        responses: List[AgentResponse],
                                        consensus_result: ConsensusResult,
                                        validation_results: Dict) -> float:
        """Calculate overall quality score for the specification"""
        
        # Component scores
        response_quality = sum(r.confidence for r in responses) / len(responses) if responses else 0
        consensus_quality = consensus_result.confidence_score
        validation_quality = 1.0 if validation_results['ready_for_consensus'] else 0.5
        
        # Weighted average
        overall_score = (response_quality * 0.4 + consensus_quality * 0.4 + validation_quality * 0.2)
        
        return min(1.0, max(0.0, overall_score))
    
    def _create_budget_exceeded_result(self, user_input: str, error_msg: str) -> SpecGenerationResult:
        """Create result when budget is exceeded"""
        return SpecGenerationResult(
            specification=f"# Budget Exceeded\n\nGeneration stopped due to budget constraints: {error_msg}",
            consensus_result=ConsensusResult(
                final_agreement="Budget exceeded during generation",
                confidence_score=0.0,
                rounds_completed=0,
                unresolved_conflicts=[],
                agent_contributions={}
            ),
            token_summary=self.token_tracker.get_session_summary(),
            discussion_log=self.discussion_log,
            generation_metadata={'budget_exceeded': True},
            quality_score=0.0
        )