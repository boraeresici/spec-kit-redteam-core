#!/usr/bin/env python3
"""
Async Multi-Agent Collaborative Specification Orchestrator

Enhanced version with proper async/await support for parallel agent communication,
concurrent execution, and improved performance.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import weakref

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

from .token_tracker import TokenTracker, BudgetExceededException
from .agents.base_agent import BaseAgent, AgentResponse, AgentResponseValidator
from .agents import AGENT_REGISTRY, DEFAULT_AGENT_CONFIGS
from .caching import SpecCache


@dataclass
class AgentDependency:
    """Represents agent execution dependencies"""
    agent_name: str
    depends_on: List[str] = field(default_factory=list)
    can_run_parallel: List[str] = field(default_factory=list)
    execution_phase: int = 1  # Lower phase numbers execute first
    estimated_duration: float = 60.0  # seconds


@dataclass
class AsyncAgentTask:
    """Represents an async agent task with progress tracking"""
    agent_name: str
    task_id: str
    prompt: str
    context: Dict[str, Any]
    priority: int = 1
    max_retries: int = 3
    timeout: float = 60.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[AgentResponse] = None
    error: Optional[Exception] = None
    dependencies: List[str] = field(default_factory=list)
    execution_phase: int = 1


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics for agents and tasks"""
    agent_name: str
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    execution_time: float = 0.0
    queue_wait_time: float = 0.0
    network_time: float = 0.0
    processing_time: float = 0.0
    tokens_processed: int = 0
    memory_usage: float = 0.0
    status: str = "running"


@dataclass
class TaskRecoveryInfo:
    """Information for task recovery and retry logic"""
    task_id: str
    retry_count: int = 0
    last_error: Optional[str] = None
    recovery_strategy: str = "retry"  # retry, fallback, skip
    fallback_agent: Optional[str] = None
    circuit_breaker_state: str = "closed"  # closed, open, half-open


@dataclass
class ConcurrentSession:
    """Manages concurrent agent communication session"""
    session_id: str
    max_concurrent_agents: int = 4
    rate_limit_delay: float = 0.1
    connection_pool_size: int = 10
    session: Optional[aiohttp.ClientSession] = None
    active_tasks: Dict[str, AsyncAgentTask] = field(default_factory=dict)
    task_queue: List[AsyncAgentTask] = field(default_factory=list)
    completed_tasks: List[AsyncAgentTask] = field(default_factory=list)
    recovery_info: Dict[str, TaskRecoveryInfo] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    performance_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_execution_time': 300.0,  # 5 minutes
        'max_queue_wait': 30.0,       # 30 seconds
        'max_memory_mb': 1000.0,      # 1GB
        'min_tokens_per_sec': 5.0     # Minimum processing speed
    })


class AsyncCollaborativeOrchestrator:
    """
    Enhanced orchestrator with async/await support for parallel agent execution
    
    Features:
    - True parallel agent communication
    - Connection pooling and rate limiting
    - Concurrent consensus building
    - Streaming response processing
    - Resource management and cleanup
    """
    
    def __init__(
        self,
        token_tracker: Optional[TokenTracker] = None,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        max_concurrent_agents: int = 4,
        connection_timeout: float = 30.0,
        console: Optional[Console] = None
    ):
        self.token_tracker = token_tracker or TokenTracker()
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.max_concurrent_agents = max_concurrent_agents
        self.connection_timeout = connection_timeout
        self.console = console or Console()
        
        # Initialize caching
        if self.enable_caching:
            self.cache = SpecCache()
        else:
            self.cache = None
        
        # Session management
        self.session: Optional[ConcurrentSession] = None
        self.progress_callback: Optional[Callable] = None
        
        # Agent dependency mapping
        self.agent_dependencies = self._initialize_agent_dependencies()
        
        # Performance tracking
        self.execution_stats = {
            'total_requests': 0,
            'parallel_requests': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'avg_response_time': 0.0,
            'dependency_optimization': 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup_session()
    
    async def _initialize_session(self):
        """Initialize HTTP session and connection pool"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_agents * 2,
            limit_per_host=self.max_concurrent_agents,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'SpecKit-RedTeam-Plugin/1.1.0'}
        )
        
        self.session = ConcurrentSession(
            session_id=f"session_{int(time.time())}",
            max_concurrent_agents=self.max_concurrent_agents,
            session=session
        )
    
    async def _cleanup_session(self):
        """Cleanup HTTP session and resources"""
        if self.session and self.session.session:
            await self.session.session.close()
            self.session = None
    
    def set_progress_callback(self, callback: Callable[[str, str, Dict], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def _initialize_agent_dependencies(self) -> Dict[str, AgentDependency]:
        """Initialize agent dependency mapping for optimal execution order"""
        
        dependencies = {
            # PM Agent - Can start immediately, provides foundation for others
            'pm': AgentDependency(
                agent_name='pm',
                depends_on=[],
                can_run_parallel=['security'],
                execution_phase=1,
                estimated_duration=45.0
            ),
            
            # Security Agent - Can run in parallel with PM
            'security': AgentDependency(
                agent_name='security',
                depends_on=[],
                can_run_parallel=['pm'],
                execution_phase=1,
                estimated_duration=50.0
            ),
            
            # Technical Agent - Depends on PM for requirements
            'technical': AgentDependency(
                agent_name='technical',
                depends_on=['pm'],
                can_run_parallel=['qa'],
                execution_phase=2,
                estimated_duration=60.0
            ),
            
            # QA Agent - Can run with Technical after PM completes
            'qa': AgentDependency(
                agent_name='qa',
                depends_on=['pm'],
                can_run_parallel=['technical'],
                execution_phase=2,
                estimated_duration=40.0
            ),
            
            # DevOps Agent - Needs both technical and security input
            'devops': AgentDependency(
                agent_name='devops',
                depends_on=['technical', 'security'],
                can_run_parallel=[],
                execution_phase=3,
                estimated_duration=35.0
            ),
            
            # Compliance Agent - Needs security foundation
            'compliance': AgentDependency(
                agent_name='compliance',
                depends_on=['security'],
                can_run_parallel=[],
                execution_phase=3,
                estimated_duration=30.0
            )
        }
        
        return dependencies
    
    def _group_agents_by_dependencies(self, agents: List[str]) -> Dict[int, List[str]]:
        """Group agents into execution phases based on dependencies"""
        
        execution_phases = {}
        
        for agent_name in agents:
            dependency = self.agent_dependencies.get(agent_name)
            if dependency:
                phase = dependency.execution_phase
                if phase not in execution_phases:
                    execution_phases[phase] = []
                execution_phases[phase].append(agent_name)
            else:
                # Unknown agent goes to phase 1
                if 1 not in execution_phases:
                    execution_phases[1] = []
                execution_phases[1].append(agent_name)
        
        return execution_phases
    
    def _validate_agent_dependencies(self, agents: List[str]) -> bool:
        """Validate that all agent dependencies are satisfied"""
        
        for agent_name in agents:
            dependency = self.agent_dependencies.get(agent_name)
            if dependency:
                # Check if all dependencies are in the agent list
                for dep in dependency.depends_on:
                    if dep not in agents:
                        self.console.print(f"⚠️  Warning: Agent '{agent_name}' depends on '{dep}' which is not in the execution list")
                        return False
        
        return True
    
    def _optimize_execution_order(self, agents: List[str]) -> List[List[str]]:
        """Optimize agent execution order based on dependencies and parallelization"""
        
        # Group by execution phases
        phases = self._group_agents_by_dependencies(agents)
        
        # Sort phases by phase number
        sorted_phases = sorted(phases.items())
        
        optimized_batches = []
        
        for phase_num, phase_agents in sorted_phases:
            # Within each phase, group parallel-compatible agents
            parallel_groups = self._create_parallel_groups(phase_agents)
            optimized_batches.extend(parallel_groups)
        
        return optimized_batches
    
    def _create_parallel_groups(self, agents: List[str]) -> List[List[str]]:
        """Create groups of agents that can run in parallel within the same phase"""
        
        if len(agents) <= 1:
            return [agents]
        
        parallel_groups = []
        remaining_agents = agents.copy()
        
        while remaining_agents:
            current_group = [remaining_agents.pop(0)]
            
            # Find agents that can run in parallel with current group
            compatible_agents = []
            for agent in remaining_agents:
                dependency = self.agent_dependencies.get(agent)
                if dependency:
                    # Check if this agent can run parallel with any agent in current group
                    can_parallel = False
                    for group_agent in current_group:
                        if (group_agent in dependency.can_run_parallel or 
                            agent in self.agent_dependencies.get(group_agent, AgentDependency("")).can_run_parallel):
                            can_parallel = True
                            break
                    
                    if can_parallel:
                        compatible_agents.append(agent)
            
            # Add compatible agents to current group (up to max concurrent limit)
            slots_available = min(self.max_concurrent_agents - len(current_group), len(compatible_agents))
            for _ in range(slots_available):
                if compatible_agents:
                    agent = compatible_agents.pop(0)
                    current_group.append(agent)
                    remaining_agents.remove(agent)
            
            parallel_groups.append(current_group)
        
        return parallel_groups
    
    def _get_fallback_agent(self, failed_agent: str) -> Optional[str]:
        """Get fallback agent for failed agent"""
        
        fallback_mapping = {
            'pm': 'technical',  # Technical can provide basic project management
            'technical': 'pm',  # PM can provide basic technical insights
            'security': 'compliance',  # Compliance has security overlap
            'compliance': 'security',  # Security can cover compliance basics
            'qa': 'technical',  # Technical can provide basic QA insights
            'devops': 'technical'  # Technical can provide basic devops insights
        }
        
        return fallback_mapping.get(failed_agent)
    
    async def _handle_task_failure(
        self, 
        task: AsyncAgentTask, 
        error: Exception
    ) -> Optional[AgentResponse]:
        """Handle task failure with recovery strategies"""
        
        if not self.session:
            return None
        
        task_recovery = self.session.recovery_info.get(
            task.task_id, 
            TaskRecoveryInfo(task_id=task.task_id)
        )
        
        task_recovery.retry_count += 1
        task_recovery.last_error = str(error)
        self.session.recovery_info[task.task_id] = task_recovery
        
        # Update error count for agent
        agent_errors = self.session.error_counts.get(task.agent_name, 0)
        self.session.error_counts[task.agent_name] = agent_errors + 1
        
        self.console.print(f"🔧 Handling failure for {task.agent_name}: {str(error)[:100]}")
        
        # Circuit breaker logic
        if agent_errors >= 3:
            task_recovery.circuit_breaker_state = "open"
            self.console.print(f"⚠️  Circuit breaker OPEN for agent {task.agent_name}")
        
        # Recovery strategies
        if task_recovery.retry_count <= task.max_retries and task_recovery.circuit_breaker_state != "open":
            # Strategy 1: Retry with exponential backoff
            backoff_delay = min(2 ** task_recovery.retry_count, 10)  # Max 10 seconds
            await asyncio.sleep(backoff_delay)
            
            self.console.print(f"🔄 Retrying {task.agent_name} (attempt {task_recovery.retry_count}/{task.max_retries})")
            
            try:
                return await self._execute_agent_request_async(task)
            except Exception as retry_error:
                self.console.print(f"🔄 Retry failed: {str(retry_error)[:100]}")
                return await self._handle_task_failure(task, retry_error)
        
        elif task_recovery.circuit_breaker_state == "open":
            # Strategy 2: Fallback agent
            fallback_agent = self._get_fallback_agent(task.agent_name)
            if fallback_agent and self.session.error_counts.get(fallback_agent, 0) < 3:
                
                self.console.print(f"🔀 Using fallback agent: {fallback_agent} for {task.agent_name}")
                
                # Create fallback task
                fallback_task = AsyncAgentTask(
                    agent_name=fallback_agent,
                    task_id=f"fallback_{task.task_id}",
                    prompt=task.prompt,
                    context=task.context,
                    priority=task.priority + 1,  # Higher priority for fallbacks
                    timeout=task.timeout
                )
                
                try:
                    return await self._execute_agent_request_async(fallback_task)
                except Exception as fallback_error:
                    self.console.print(f"🔀 Fallback also failed: {str(fallback_error)[:100]}")
        
        # Strategy 3: Degraded response
        return await self._generate_degraded_response(task, task_recovery)
    
    async def _generate_degraded_response(
        self, 
        task: AsyncAgentTask, 
        recovery_info: TaskRecoveryInfo
    ) -> AgentResponse:
        """Generate a degraded response when all recovery strategies fail"""
        
        degraded_content = f"""
        [DEGRADED RESPONSE - Agent {task.agent_name} unavailable]
        
        Based on the request: {task.context.get('description', 'N/A')}
        
        This is a fallback response due to agent failure. Key considerations:
        
        - Original agent ({task.agent_name}) experienced technical difficulties
        - Retry attempts: {recovery_info.retry_count}
        - Last error: {recovery_info.last_error[:200] if recovery_info.last_error else 'Unknown'}
        
        Recommended actions:
        1. Review the specific requirements for {task.agent_name} perspective
        2. Consider manual input for critical {task.agent_name} requirements
        3. Verify system availability for future requests
        
        This degraded response ensures continuity while maintaining transparency about limitations.
        """
        
        return AgentResponse(
            agent_name=task.agent_name,
            content=degraded_content,
            confidence_score=0.3,  # Low confidence for degraded responses
            processing_time=0.0,
            tokens_used=0,
            metadata={
                'degraded': True,
                'original_agent': task.agent_name,
                'failure_reason': recovery_info.last_error,
                'retry_attempts': recovery_info.retry_count
            }
        )
    
    def _start_performance_monitoring(self, task: AsyncAgentTask) -> PerformanceMetrics:
        """Start performance monitoring for a task"""
        
        if not self.session:
            return None
        
        metrics = PerformanceMetrics(
            agent_name=task.agent_name,
            task_id=task.task_id,
            start_time=time.time(),
            status="running"
        )
        
        self.session.performance_metrics[task.task_id] = metrics
        return metrics
    
    def _finish_performance_monitoring(
        self, 
        task: AsyncAgentTask, 
        response: Optional[AgentResponse] = None
    ):
        """Finish performance monitoring and check for alerts"""
        
        if not self.session or task.task_id not in self.session.performance_metrics:
            return
        
        metrics = self.session.performance_metrics[task.task_id]
        metrics.end_time = time.time()
        metrics.execution_time = metrics.end_time - metrics.start_time
        metrics.status = "completed" if response else "failed"
        
        if response:
            metrics.tokens_processed = getattr(response, 'tokens_used', 0)
            metrics.processing_time = getattr(response, 'processing_time', 0.0)
        
        # Calculate tokens per second
        if metrics.execution_time > 0 and metrics.tokens_processed > 0:
            tokens_per_sec = metrics.tokens_processed / metrics.execution_time
        else:
            tokens_per_sec = 0
        
        # Check for performance alerts
        self._check_performance_alerts(metrics, tokens_per_sec)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics, tokens_per_sec: float):
        """Check performance metrics against thresholds and trigger alerts"""
        
        if not self.session:
            return
        
        thresholds = self.session.alert_thresholds
        alerts = []
        
        # Execution time alert
        if metrics.execution_time > thresholds['max_execution_time']:
            alerts.append({
                'type': 'SLOW_EXECUTION',
                'agent': metrics.agent_name,
                'task_id': metrics.task_id,
                'value': metrics.execution_time,
                'threshold': thresholds['max_execution_time'],
                'message': f"Agent {metrics.agent_name} took {metrics.execution_time:.1f}s (threshold: {thresholds['max_execution_time']}s)"
            })
        
        # Queue wait time alert
        if metrics.queue_wait_time > thresholds['max_queue_wait']:
            alerts.append({
                'type': 'HIGH_QUEUE_WAIT',
                'agent': metrics.agent_name,
                'task_id': metrics.task_id,
                'value': metrics.queue_wait_time,
                'threshold': thresholds['max_queue_wait'],
                'message': f"Agent {metrics.agent_name} waited {metrics.queue_wait_time:.1f}s in queue"
            })
        
        # Memory usage alert
        if metrics.memory_usage > thresholds['max_memory_mb']:
            alerts.append({
                'type': 'HIGH_MEMORY',
                'agent': metrics.agent_name,
                'task_id': metrics.task_id,
                'value': metrics.memory_usage,
                'threshold': thresholds['max_memory_mb'],
                'message': f"Agent {metrics.agent_name} used {metrics.memory_usage:.1f}MB memory"
            })
        
        # Low processing speed alert
        if tokens_per_sec > 0 and tokens_per_sec < thresholds['min_tokens_per_sec']:
            alerts.append({
                'type': 'SLOW_PROCESSING',
                'agent': metrics.agent_name,
                'task_id': metrics.task_id,
                'value': tokens_per_sec,
                'threshold': thresholds['min_tokens_per_sec'],
                'message': f"Agent {metrics.agent_name} processing {tokens_per_sec:.1f} tokens/sec (slow)"
            })
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger performance alert"""
        
        alert_icons = {
            'SLOW_EXECUTION': '🐌',
            'HIGH_QUEUE_WAIT': '⏱️',
            'HIGH_MEMORY': '💾',
            'SLOW_PROCESSING': '⚡'
        }
        
        icon = alert_icons.get(alert['type'], '⚠️')
        self.console.print(f"{icon} [yellow]ALERT[/yellow] - {alert['message']}")
        
        # Update execution stats with alert
        if 'alerts' not in self.execution_stats:
            self.execution_stats['alerts'] = []
        
        self.execution_stats['alerts'].append({
            'timestamp': time.time(),
            'type': alert['type'],
            'agent': alert['agent'],
            'value': alert['value'],
            'threshold': alert['threshold']
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        if not self.session:
            return {}
        
        metrics_list = list(self.session.performance_metrics.values())
        if not metrics_list:
            return {}
        
        # Calculate aggregated metrics
        completed_metrics = [m for m in metrics_list if m.end_time is not None]
        
        if not completed_metrics:
            return {'status': 'no_completed_tasks'}
        
        execution_times = [m.execution_time for m in completed_metrics]
        processing_times = [m.processing_time for m in completed_metrics]
        tokens_processed = [m.tokens_processed for m in completed_metrics]
        
        return {
            'total_tasks': len(completed_metrics),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'total_tokens_processed': sum(tokens_processed),
            'avg_tokens_per_second': (
                sum(tokens_processed) / sum(execution_times) 
                if sum(execution_times) > 0 else 0
            ),
            'alerts_triggered': len(self.execution_stats.get('alerts', [])),
            'agent_performance': self._get_agent_performance_breakdown(),
            'dependency_optimization_time': self.execution_stats.get('dependency_optimization', 0.0)
        }
    
    def _get_agent_performance_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by agent"""
        
        if not self.session:
            return {}
        
        agent_metrics = {}
        
        for metrics in self.session.performance_metrics.values():
            if metrics.end_time is None:
                continue
            
            agent = metrics.agent_name
            if agent not in agent_metrics:
                agent_metrics[agent] = {
                    'total_execution_time': 0.0,
                    'total_tokens': 0,
                    'task_count': 0,
                    'avg_execution_time': 0.0,
                    'tokens_per_second': 0.0
                }
            
            agent_metrics[agent]['total_execution_time'] += metrics.execution_time
            agent_metrics[agent]['total_tokens'] += metrics.tokens_processed
            agent_metrics[agent]['task_count'] += 1
        
        # Calculate averages
        for agent, stats in agent_metrics.items():
            if stats['task_count'] > 0:
                stats['avg_execution_time'] = stats['total_execution_time'] / stats['task_count']
            
            if stats['total_execution_time'] > 0:
                stats['tokens_per_second'] = stats['total_tokens'] / stats['total_execution_time']
        
        return agent_metrics
    
    def _update_progress(self, phase: str, details: str, data: Dict = None):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback(phase, details, data or {})
    
    async def generate_collaborative_spec(
        self,
        description: str,
        agents: List[str],
        max_budget: float,
        complexity: str = 'medium',
        enable_streaming: bool = True
    ) -> 'SpecGenerationResult':
        """
        Generate specification using collaborative AI with async/await
        
        Args:
            description: Feature/system description
            agents: List of agent types to use
            max_budget: Maximum budget for generation
            complexity: Complexity level (simple/medium/complex)
            enable_streaming: Enable streaming responses
        
        Returns:
            SpecGenerationResult with complete specification
        """
        
        start_time = time.time()
        self._update_progress("initialization", "Starting collaborative generation")
        
        try:
            # Phase 1: Parallel initial analysis
            initial_responses = await self._parallel_initial_analysis(
                description, agents, complexity
            )
            
            self._update_progress("initial_analysis", "Completed initial analysis", {
                'agents_completed': len(initial_responses),
                'total_agents': len(agents)
            })
            
            # Phase 2: Cross-pollination and refinement
            refined_responses = await self._parallel_cross_pollination(
                initial_responses, description
            )
            
            self._update_progress("refinement", "Completed cross-pollination")
            
            # Phase 3: Async consensus building
            consensus_result = await self._async_consensus_building(
                refined_responses, description
            )
            
            self._update_progress("consensus", "Built consensus")
            
            # Phase 4: Final specification generation
            final_spec = await self._generate_final_specification(
                consensus_result, initial_responses, description
            )
            
            # Compile results
            generation_time = time.time() - start_time
            self.execution_stats['total_time'] = generation_time
            
            result = SpecGenerationResult(
                specification=final_spec,
                consensus_result=consensus_result,
                token_summary=self.token_tracker.get_summary(),
                discussion_log=self._compile_discussion_log(),
                generation_metadata={
                    'generation_time': generation_time,
                    'agents_used': agents,
                    'parallel_execution': self.enable_parallel,
                    'execution_stats': self.execution_stats
                },
                quality_score=self._calculate_quality_score(consensus_result)
            )
            
            self._update_progress("completed", "Generation completed successfully")
            return result
            
        except BudgetExceededException:
            self._update_progress("budget_exceeded", "Budget limit reached")
            raise
        except Exception as e:
            self._update_progress("error", f"Generation failed: {str(e)}")
            raise
    
    async def _parallel_initial_analysis(
        self,
        description: str,
        agents: List[str],
        complexity: str
    ) -> Dict[str, AgentResponse]:
        """Run initial analysis with dependency-optimized agent execution"""
        
        start_time = time.time()
        self._update_progress("initial_analysis", "Optimizing agent execution order")
        
        # Validate dependencies
        if not self._validate_agent_dependencies(agents):
            self.console.print("⚠️  Proceeding with standard parallel execution due to dependency issues")
        
        # Optimize execution order based on dependencies
        execution_batches = self._optimize_execution_order(agents)
        
        self._update_progress("dependency_optimization", f"Optimized into {len(execution_batches)} execution batches")
        
        # Execute in optimized batches
        all_responses = {}
        
        for batch_idx, agent_batch in enumerate(execution_batches):
            self._update_progress(
                "execution_batch", 
                f"Executing batch {batch_idx + 1}/{len(execution_batches)}: {', '.join(agent_batch)}"
            )
            
            # Create tasks for current batch
            batch_tasks = []
            for agent_name in agent_batch:
                # Get dependency info for timeout adjustment
                dependency = self.agent_dependencies.get(agent_name)
                timeout = dependency.estimated_duration if dependency else 60.0
                
                task = AsyncAgentTask(
                    agent_name=agent_name,
                    task_id=f"initial_{agent_name}",
                    prompt=self._create_initial_prompt(description, agent_name, complexity),
                    context={
                        'phase': 'initial', 
                        'description': description, 
                        'complexity': complexity,
                        'previous_responses': all_responses  # Provide context from previous batches
                    },
                    priority=1,
                    timeout=timeout,
                    execution_phase=batch_idx + 1
                )
                batch_tasks.append(task)
            
            # Execute current batch
            if self.enable_parallel and len(agent_batch) > 1:
                batch_responses = await self._execute_parallel_tasks(batch_tasks)
            else:
                batch_responses = await self._execute_sequential_tasks(batch_tasks)
            
            # Merge responses
            all_responses.update(batch_responses)
        
        optimization_time = time.time() - start_time
        self.execution_stats['dependency_optimization'] = optimization_time
        
        self._update_progress(
            "initial_analysis", 
            f"Completed dependency-optimized execution in {optimization_time:.2f}s"
        )
        
        return all_responses
    
    async def _parallel_cross_pollination(
        self,
        initial_responses: Dict[str, AgentResponse],
        description: str
    ) -> Dict[str, AgentResponse]:
        """Cross-pollination phase with parallel execution"""
        
        self._update_progress("cross_pollination", "Starting cross-pollination")
        
        agent_tasks = []
        
        for agent_name, response in initial_responses.items():
            # Get other agents' insights for cross-pollination
            other_insights = {
                name: resp.content for name, resp in initial_responses.items()
                if name != agent_name
            }
            
            task = AsyncAgentTask(
                agent_name=agent_name,
                task_id=f"cross_{agent_name}",
                prompt=self._create_cross_pollination_prompt(
                    description, response.content, other_insights
                ),
                context={
                    'phase': 'cross_pollination',
                    'original_response': response,
                    'other_insights': other_insights
                },
                priority=2,
                timeout=90.0
            )
            agent_tasks.append(task)
        
        return await self._execute_parallel_tasks(agent_tasks)
    
    async def _async_consensus_building(
        self,
        responses: Dict[str, AgentResponse],
        description: str
    ) -> 'ConsensusResult':
        """Build consensus asynchronously with conflict resolution"""
        
        self._update_progress("consensus", "Building consensus")
        
        # Identify conflicts concurrently
        conflicts = await self._identify_conflicts_async(responses)
        
        # Resolve conflicts in parallel
        resolved_conflicts = await self._resolve_conflicts_parallel(
            conflicts, responses, description
        )
        
        # Build final consensus
        final_agreement = await self._build_final_agreement(
            responses, resolved_conflicts
        )
        
        confidence_score = self._calculate_confidence_score(
            responses, resolved_conflicts
        )
        
        return ConsensusResult(
            final_agreement=final_agreement,
            confidence_score=confidence_score,
            rounds_completed=len(resolved_conflicts) + 1,
            unresolved_conflicts=[c for c in conflicts if c.resolution_strategy == 'unresolved'],
            agent_contributions=self._calculate_agent_contributions(responses)
        )
    
    async def _execute_parallel_tasks(
        self,
        tasks: List[AsyncAgentTask]
    ) -> Dict[str, AgentResponse]:
        """Execute multiple agent tasks in parallel with rate limiting"""
        
        if not self.session:
            await self._initialize_session()
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        
        async def execute_task(task: AsyncAgentTask) -> Tuple[str, Optional[AgentResponse]]:
            async with semaphore:
                # Start performance monitoring
                metrics = self._start_performance_monitoring(task)
                
                try:
                    # Rate limiting
                    await asyncio.sleep(self.session.rate_limit_delay)
                    
                    task.started_at = datetime.now()
                    task.status = "running"
                    
                    self._update_progress("agent_task", f"Executing {task.agent_name}", {
                        'task_id': task.task_id,
                        'agent': task.agent_name
                    })
                    
                    # Check cache first
                    if self.cache:
                        cached_response = await self._check_cache_async(task)
                        if cached_response:
                            self.execution_stats['cache_hits'] += 1
                            task.result = cached_response
                            task.status = "completed"
                            task.completed_at = datetime.now()
                            
                            # Finish monitoring for cached response
                            self._finish_performance_monitoring(task, cached_response)
                            return task.agent_name, cached_response
                    
                    # Execute agent request
                    response = await self._execute_agent_request_async(task)
                    
                    # Cache response
                    if self.cache and response:
                        await self._cache_response_async(task, response)
                    
                    task.result = response
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    self.execution_stats['total_requests'] += 1
                    
                    # Finish monitoring for successful response
                    self._finish_performance_monitoring(task, response)
                    return task.agent_name, response
                    
                except Exception as e:
                    task.error = e
                    task.status = "failed"
                    task.completed_at = datetime.now()
                    
                    self.console.print(f"❌ Task {task.task_id} failed: {str(e)[:100]}")
                    
                    # Attempt recovery
                    recovery_response = await self._handle_task_failure(task, e)
                    if recovery_response:
                        task.result = recovery_response
                        task.status = "recovered"
                        self.console.print(f"🔧 Task {task.task_id} recovered successfully")
                        
                        # Finish monitoring for recovered response
                        self._finish_performance_monitoring(task, recovery_response)
                        return task.agent_name, recovery_response
                    else:
                        # Finish monitoring for failed response
                        self._finish_performance_monitoring(task, None)
                        return task.agent_name, None
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Filter successful results
        responses = {}
        for result in results:
            if isinstance(result, tuple):
                agent_name, response = result
                if response:
                    responses[agent_name] = response
        
        self.execution_stats['parallel_requests'] += len(tasks)
        return responses
    
    async def _execute_sequential_tasks(
        self,
        tasks: List[AsyncAgentTask]
    ) -> Dict[str, AgentResponse]:
        """Execute tasks sequentially (fallback when parallel is disabled)"""
        
        responses = {}
        
        for task in tasks:
            try:
                self._update_progress("agent_task", f"Executing {task.agent_name}")
                
                # Check cache
                if self.cache:
                    cached_response = await self._check_cache_async(task)
                    if cached_response:
                        responses[task.agent_name] = cached_response
                        continue
                
                # Execute request
                response = await self._execute_agent_request_async(task)
                if response:
                    responses[task.agent_name] = response
                    
                    # Cache response
                    if self.cache:
                        await self._cache_response_async(task, response)
                
            except Exception as e:
                self.console.print(f"❌ Sequential task {task.task_id} failed: {e}")
                continue
        
        return responses
    
    async def _execute_agent_request_async(
        self,
        task: AsyncAgentTask
    ) -> Optional[AgentResponse]:
        """Execute individual agent request asynchronously"""
        
        # Get agent class
        agent_class = AGENT_REGISTRY.get(task.agent_name)
        if not agent_class:
            raise ValueError(f"Unknown agent: {task.agent_name}")
        
        # Get agent config
        config = DEFAULT_AGENT_CONFIGS.get(task.agent_name, {})
        
        # Create agent instance
        agent = agent_class(
            model=config.get('model', 'gpt-4'),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 2000),
            token_tracker=self.token_tracker
        )
        
        # Execute request with timeout
        try:
            response = await asyncio.wait_for(
                agent.process_async(task.prompt, task.context),
                timeout=task.timeout
            )
            return response
            
        except asyncio.TimeoutError:
            raise Exception(f"Agent {task.agent_name} request timed out after {task.timeout}s")
    
    async def _check_cache_async(self, task: AsyncAgentTask) -> Optional[AgentResponse]:
        """Check cache for existing response asynchronously"""
        if not self.cache:
            return None
        
        cache_key = f"{task.agent_name}:{hash(task.prompt)}"
        return await asyncio.get_event_loop().run_in_executor(
            None, self.cache.get_response, cache_key
        )
    
    async def _cache_response_async(self, task: AsyncAgentTask, response: AgentResponse):
        """Cache response asynchronously"""
        if not self.cache:
            return
        
        cache_key = f"{task.agent_name}:{hash(task.prompt)}"
        await asyncio.get_event_loop().run_in_executor(
            None, self.cache.store_response, cache_key, response
        )
    
    async def _identify_conflicts_async(
        self,
        responses: Dict[str, AgentResponse]
    ) -> List['ConflictItem']:
        """Identify conflicts between agent responses asynchronously"""
        
        conflicts = []
        agent_names = list(responses.keys())
        
        # Compare all pairs of agents concurrently
        comparison_tasks = []
        
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent_1, agent_2 = agent_names[i], agent_names[j]
                comparison_tasks.append(
                    self._compare_responses_async(
                        agent_1, responses[agent_1],
                        agent_2, responses[agent_2]
                    )
                )
        
        # Execute comparisons in parallel
        comparison_results = await asyncio.gather(*comparison_tasks, return_exceptions=True)
        
        # Collect conflicts
        for result in comparison_results:
            if isinstance(result, list):
                conflicts.extend(result)
        
        return conflicts
    
    async def _compare_responses_async(
        self,
        agent_1: str, response_1: AgentResponse,
        agent_2: str, response_2: AgentResponse
    ) -> List['ConflictItem']:
        """Compare two agent responses for conflicts asynchronously"""
        
        # This would use NLP/AI to identify semantic conflicts
        # For now, simple keyword-based detection
        conflicts = []
        
        # Example conflict detection logic
        content_1 = response_1.content.lower()
        content_2 = response_2.content.lower()
        
        # Technology stack conflicts
        if ('microservices' in content_1 and 'monolith' in content_2) or \
           ('monolith' in content_1 and 'microservices' in content_2):
            conflicts.append(ConflictItem(
                agent_1=agent_1,
                agent_2=agent_2,
                conflict_type='architecture',
                description='Disagreement on microservices vs monolithic architecture',
                resolution_strategy='needs_resolution',
                severity='high'
            ))
        
        # Database conflicts
        db_keywords_1 = set(['mysql', 'postgresql', 'mongodb', 'redis']) & set(content_1.split())
        db_keywords_2 = set(['mysql', 'postgresql', 'mongodb', 'redis']) & set(content_2.split())
        
        if db_keywords_1 and db_keywords_2 and db_keywords_1 != db_keywords_2:
            conflicts.append(ConflictItem(
                agent_1=agent_1,
                agent_2=agent_2,
                conflict_type='database',
                description=f'Database preference conflict: {db_keywords_1} vs {db_keywords_2}',
                resolution_strategy='needs_resolution',
                severity='medium'
            ))
        
        return conflicts
    
    async def _resolve_conflicts_parallel(
        self,
        conflicts: List['ConflictItem'],
        responses: Dict[str, AgentResponse],
        description: str
    ) -> List['ConflictItem']:
        """Resolve conflicts in parallel"""
        
        if not conflicts:
            return []
        
        resolution_tasks = []
        
        for conflict in conflicts:
            resolution_tasks.append(
                self._resolve_single_conflict_async(conflict, responses, description)
            )
        
        resolved_conflicts = await asyncio.gather(*resolution_tasks, return_exceptions=True)
        
        return [c for c in resolved_conflicts if isinstance(c, ConflictItem)]
    
    async def _resolve_single_conflict_async(
        self,
        conflict: 'ConflictItem',
        responses: Dict[str, AgentResponse],
        description: str
    ) -> 'ConflictItem':
        """Resolve a single conflict asynchronously"""
        
        # Create a resolution prompt
        resolution_prompt = f"""
        Conflict Resolution Required:
        
        Description: {description}
        
        Conflict: {conflict.description}
        Agent 1 ({conflict.agent_1}) perspective: {responses[conflict.agent_1].content[:500]}...
        Agent 2 ({conflict.agent_2}) perspective: {responses[conflict.agent_2].content[:500]}...
        
        Provide a balanced resolution that considers both perspectives.
        """
        
        # Use a neutral resolver (could be a separate agent)
        resolution_task = AsyncAgentTask(
            agent_name='technical',  # Use technical agent as resolver
            task_id=f"resolve_{conflict.conflict_type}",
            prompt=resolution_prompt,
            context={'phase': 'conflict_resolution', 'conflict': conflict}
        )
        
        try:
            resolution_response = await self._execute_agent_request_async(resolution_task)
            if resolution_response:
                conflict.resolution_strategy = 'resolved'
                conflict.description += f" | Resolution: {resolution_response.content[:200]}..."
        except Exception:
            conflict.resolution_strategy = 'unresolved'
        
        return conflict
    
    async def _build_final_agreement(
        self,
        responses: Dict[str, AgentResponse],
        resolved_conflicts: List['ConflictItem']
    ) -> str:
        """Build final agreement incorporating all perspectives"""
        
        agreement_sections = []
        
        # Combine agent responses
        for agent_name, response in responses.items():
            section = f"\n## {agent_name.title()} Agent Perspective\n{response.content}"
            agreement_sections.append(section)
        
        # Add conflict resolutions
        if resolved_conflicts:
            resolution_section = "\n## Conflict Resolutions\n"
            for conflict in resolved_conflicts:
                if conflict.resolution_strategy == 'resolved':
                    resolution_section += f"- {conflict.description}\n"
            agreement_sections.append(resolution_section)
        
        return '\n'.join(agreement_sections)
    
    async def _generate_final_specification(
        self,
        consensus_result: 'ConsensusResult',
        initial_responses: Dict[str, AgentResponse],
        description: str
    ) -> str:
        """Generate final specification document"""
        
        self._update_progress("final_spec", "Generating final specification")
        
        # Create comprehensive specification prompt
        spec_prompt = f"""
        Generate a comprehensive technical specification based on the collaborative analysis:
        
        Original Request: {description}
        
        Collaborative Analysis:
        {consensus_result.final_agreement}
        
        Please create a well-structured specification document including:
        1. Executive Summary
        2. Requirements
        3. Technical Architecture
        4. Security Considerations
        5. Testing Strategy
        6. Implementation Plan
        """
        
        # Use the highest-contributing agent to generate final spec
        best_agent = max(
            consensus_result.agent_contributions.items(),
            key=lambda x: x[1]
        )[0]
        
        spec_task = AsyncAgentTask(
            agent_name=best_agent,
            task_id="final_specification",
            prompt=spec_prompt,
            context={'phase': 'final_generation', 'consensus': consensus_result}
        )
        
        spec_response = await self._execute_agent_request_async(spec_task)
        
        if spec_response:
            return spec_response.content
        else:
            # Fallback to consensus agreement
            return consensus_result.final_agreement
    
    def _create_initial_prompt(self, description: str, agent_name: str, complexity: str) -> str:
        """Create initial analysis prompt for agent"""
        return f"Analyze this {complexity} complexity request from your {agent_name} perspective: {description}"
    
    def _create_cross_pollination_prompt(
        self,
        description: str,
        original_response: str,
        other_insights: Dict[str, str]
    ) -> str:
        """Create cross-pollination prompt"""
        other_perspectives = '\n'.join([
            f"{agent}: {insight[:200]}..." for agent, insight in other_insights.items()
        ])
        
        return f"""
        Refine your analysis considering other perspectives:
        
        Original request: {description}
        Your original response: {original_response[:300]}...
        
        Other agent perspectives:
        {other_perspectives}
        
        Provide a refined analysis that incorporates valuable insights from other agents.
        """
    
    def _calculate_confidence_score(
        self,
        responses: Dict[str, AgentResponse],
        conflicts: List['ConflictItem']
    ) -> float:
        """Calculate confidence score based on consensus and conflicts"""
        
        if not responses:
            return 0.0
        
        # Base score from number of agents
        base_score = min(len(responses) * 0.2, 0.8)
        
        # Penalty for unresolved conflicts
        unresolved_conflicts = sum(1 for c in conflicts if c.resolution_strategy == 'unresolved')
        conflict_penalty = unresolved_conflicts * 0.1
        
        return max(0.0, min(1.0, base_score - conflict_penalty))
    
    def _calculate_agent_contributions(
        self,
        responses: Dict[str, AgentResponse]
    ) -> Dict[str, float]:
        """Calculate relative contribution of each agent"""
        
        if not responses:
            return {}
        
        # Simple equal weighting for now
        weight = 1.0 / len(responses)
        return {agent: weight for agent in responses.keys()}
    
    def _calculate_quality_score(self, consensus_result: 'ConsensusResult') -> float:
        """Calculate overall quality score"""
        return consensus_result.confidence_score
    
    def _compile_discussion_log(self) -> List[Dict]:
        """Compile discussion log from session"""
        if not self.session:
            return []
        
        log = []
        for task in self.session.completed_tasks:
            log.append({
                'agent': task.agent_name,
                'task_id': task.task_id,
                'status': task.status,
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'duration': (
                    (task.completed_at - task.started_at).total_seconds()
                    if task.started_at and task.completed_at else None
                )
            })
        
        return log


# Import necessary dataclasses from original orchestrator
from .orchestrator import ConflictItem, ConsensusResult, SpecGenerationResult