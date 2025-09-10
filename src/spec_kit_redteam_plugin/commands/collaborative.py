#!/usr/bin/env python3
"""
Collaborative AI specification generation commands.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.text import Text
from rich.align import Align

from ..token_tracker import TokenTracker, BudgetExceededException
from ..orchestrator import CollaborativeSpecOrchestrator, SpecGenerationResult
from ..async_orchestrator import AsyncCollaborativeOrchestrator
from ..agents import AGENT_REGISTRY, DEFAULT_AGENT_CONFIGS
from ..budget_ui import BudgetManager
from ..error_handler import error_handler, PluginError, user_input_error, api_error
from ..version_manager import version_manager
from ..recovery_manager import recovery_manager, with_recovery


app = typer.Typer(name="collab", help="Multi-Agent Collaborative AI")
console = Console()


def _validate_generate_inputs(description: str, agents: List[str], budget: float, complexity: str):
    """Validate generate command inputs with user-friendly error messages"""
    
    # Validate description
    if not description or description.strip() == "":
        raise user_input_error(
            "Description cannot be empty",
            suggestions=[
                "Provide a clear description of what you want to build",
                "Example: 'Build a secure REST API for user management'",
                "Be specific about requirements and constraints"
            ]
        )
    
    if len(description.strip()) < 10:
        raise user_input_error(
            "Description is too short (minimum 10 characters)",
            suggestions=[
                "Provide more detail about your requirements",
                "Include key features and functionality",
                "Mention any specific constraints or technologies"
            ]
        )
    
    # Validate agents
    if not agents:
        raise user_input_error(
            "At least one agent must be specified",
            suggestions=[
                "Use --agent pm for product management analysis",
                "Use --agent technical for technical architecture",
                "Use --agent security for security analysis",
                "Use --agent qa for quality assurance testing"
            ]
        )
    
    invalid_agents = [a for a in agents if a not in AGENT_REGISTRY]
    if invalid_agents:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise user_input_error(
            f"Invalid agent(s): {', '.join(invalid_agents)}",
            suggestions=[
                f"Available agents: {available}",
                "Use 'specify collab agents' to see agent details",
                "Check spelling of agent names"
            ]
        )
    
    # Validate budget
    if budget <= 0:
        raise user_input_error(
            "Budget must be greater than 0",
            suggestions=[
                "Set a reasonable budget (e.g., --budget 25.00)",
                "Minimum recommended budget: $5.00",
                "Complex specifications may require $20-50"
            ]
        )
    
    if budget < 2.0:
        raise user_input_error(
            f"Budget ${budget:.2f} is very low and may not complete generation",
            suggestions=[
                "Recommended minimum: $5.00",
                "Simple specs: $5-15",
                "Complex specs: $20-50",
                "Use --budget <amount> to set higher budget"
            ]
        )
    
    # Validate complexity
    valid_complexity = ['simple', 'medium', 'complex']
    if complexity not in valid_complexity:
        raise user_input_error(
            f"Invalid complexity: {complexity}",
            suggestions=[
                f"Valid options: {', '.join(valid_complexity)}",
                "simple: Basic features, minimal requirements",
                "medium: Standard features with some complexity",
                "complex: Advanced features, high requirements"
            ]
        )


class LiveProgressTracker:
    """Live progress tracking for collaborative generation"""
    
    def __init__(self, console: Console):
        self.console = console
        self.start_time = time.time()
        self.current_phase = "Initializing"
        self.agents_status = {}
        self.token_usage = {}
        
    def update_phase(self, phase: str):
        self.current_phase = phase
        
    def update_agent_status(self, agent_name: str, status: str, details: str = ""):
        self.agents_status[agent_name] = {
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
    
    def update_token_usage(self, token_summary: Dict):
        self.token_usage = token_summary
    
    def create_status_panel(self) -> Panel:
        """Create current status panel"""
        elapsed = time.time() - self.start_time
        
        # Phase info
        phase_text = Text()
        phase_text.append(f"🔄 Current Phase: ", style="bold")
        phase_text.append(self.current_phase, style="cyan")
        phase_text.append(f" ({elapsed:.1f}s)", style="dim")
        
        content = [phase_text]
        
        # Agent status
        if self.agents_status:
            content.append(Text("\n🤖 Agent Status:", style="bold"))
            for agent, status_info in self.agents_status.items():
                status_line = Text()
                
                if status_info['status'] == 'completed':
                    status_line.append("  ✅ ", style="green")
                elif status_info['status'] == 'working':
                    status_line.append("  🔄 ", style="cyan")
                elif status_info['status'] == 'pending':
                    status_line.append("  ⏳ ", style="yellow")
                else:
                    status_line.append("  ❌ ", style="red")
                
                status_line.append(f"{agent}: ", style="white")
                status_line.append(status_info['details'], style="dim")
                content.append(status_line)
        
        # Token usage
        if self.token_usage:
            content.append(Text(f"\n💰 Usage: {self.token_usage.get('total_tokens', 0):,} tokens, ${self.token_usage.get('total_cost', 0):.2f}", style="green"))
            
            if 'budget_limit' in self.token_usage and self.token_usage['budget_limit']:
                usage_pct = self.token_usage.get('budget_used_percentage', 0)
                content.append(Text(f"📊 Budget: {usage_pct:.1f}% used", 
                                  style="yellow" if usage_pct > 70 else "green"))
        
        combined_content = Text()
        for item in content:
            combined_content.append_text(item)
            combined_content.append("\n")
        
        return Panel(
            combined_content,
            title="[bold cyan]Collaborative Generation Status[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )


class BudgetManager:
    """Manages budget display and confirmation"""
    
    def __init__(self, console: Console):
        self.console = console
    
    def show_cost_estimate(self, 
                          agents: List[str],
                          complexity: str = 'medium',
                          budget: float = 50.0) -> bool:
        """Show cost estimate and get user confirmation"""
        
        # Cost estimates based on agent types and complexity
        base_costs = {
            'pm': {'simple': 0.30, 'medium': 0.60, 'complex': 1.20},
            'technical': {'simple': 0.40, 'medium': 0.80, 'complex': 1.60},
            'security': {'simple': 0.35, 'medium': 0.70, 'complex': 1.40},
            'qa': {'simple': 0.20, 'medium': 0.40, 'complex': 0.80}
        }
        
        consensus_cost = {'simple': 0.50, 'medium': 1.00, 'complex': 2.00}
        
        table = Table(title="💰 Collaborative Generation - Cost Estimate")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Estimated Cost", justify="right", style="green", width=15)
        table.add_column("Notes", style="dim", width=30)
        
        total_estimated = 0
        
        for agent in agents:
            agent_cost = base_costs.get(agent, base_costs['pm'])[complexity]
            total_estimated += agent_cost
            agent_name = agent.title() + " Agent"
            model = DEFAULT_AGENT_CONFIGS.get(agent, {}).get('model', 'gpt-4')
            table.add_row(agent_name, f"${agent_cost:.2f}", f"Using {model}")
        
        consensus_estimated = consensus_cost[complexity]
        total_estimated += consensus_estimated
        table.add_row("Consensus Building", f"${consensus_estimated:.2f}", f"{len(agents)} agent discussion")
        
        table.add_row("", "", "", style="dim")
        table.add_row("Total Estimated", f"${total_estimated:.2f}", 
                     f"Budget: ${budget:.2f}", style="bold green")
        
        self.console.print(table)
        self.console.print()
        
        if total_estimated > budget:
            self.console.print(f"⚠️  [yellow]Estimated cost (${total_estimated:.2f}) exceeds budget (${budget:.2f})[/yellow]")
            self.console.print("Consider reducing agents or increasing budget.")
            return typer.confirm("Proceed anyway?")
        
        return typer.confirm(f"Proceed with estimated cost of ${total_estimated:.2f}?")
    
    def show_final_summary(self, token_summary: Dict):
        """Show final cost and usage summary"""
        summary_table = Table(title="📊 Generation Complete - Final Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="white")
        
        summary_table.add_row("Total Tokens", f"{token_summary.get('total_tokens', 0):,}")
        summary_table.add_row("Total Cost", f"${token_summary.get('total_cost', 0):.2f}")
        summary_table.add_row("Duration", f"{token_summary.get('duration_seconds', 0):,}s")
        summary_table.add_row("Operations", f"{token_summary.get('operation_count', 0)}")
        
        if token_summary.get('budget_limit'):
            usage_pct = token_summary.get('budget_used_percentage', 0)
            summary_table.add_row("Budget Used", f"{usage_pct:.1f}%")
        
        self.console.print(summary_table)


@app.command("generate")
async def generate_collaborative_spec(
    description: str = typer.Argument(..., help="Feature description"),
    agents: List[str] = typer.Option(["pm", "technical"], "--agent", "-a", help="Agents to include"),
    budget: float = typer.Option(50.0, "--budget", "-b", help="Session budget in USD"),
    complexity: str = typer.Option("medium", "--complexity", "-c", help="Complexity level (simple/medium/complex)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Enable parallel agent execution"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable response caching"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate specification using collaborative AI agents"""
    
    try:
        # Input validation with enhanced error messages
        _validate_generate_inputs(description, agents, budget, complexity)
        
    except PluginError as e:
        if not error_handler.handle_error(e, {"command": "generate"}):
            raise typer.Exit(1)
        return
    
    # Budget confirmation
    budget_manager = BudgetManager(console)
    if not budget_manager.show_cost_estimate(agents, complexity, budget):
        console.print("[yellow]Generation cancelled[/yellow]")
        raise typer.Exit(0)
    
    console.print()  # Blank line
    
    # Initialize token tracker
    token_tracker = TokenTracker(session_budget=budget)
    
    # Initialize async orchestrator for better performance
    if parallel:
        orchestrator = AsyncCollaborativeOrchestrator(
            token_tracker=token_tracker,
            enable_caching=cache,
            enable_parallel=parallel,
            max_concurrent_agents=min(len(agents), 4),  # Optimize based on agent count
            console=console
        )
        use_async = True
    else:
        # Fallback to synchronous orchestrator
        orchestrator = CollaborativeSpecOrchestrator(
            token_tracker=token_tracker,
            enable_caching=cache,
            enable_parallel=False
        )
        use_async = False
    
    # Progress tracking
    progress_tracker = LiveProgressTracker(console)
    
    try:
        # Live progress display
        with Live(progress_tracker.create_status_panel(), 
                 console=console, 
                 refresh_per_second=2,
                 transient=True) as live:
            
            # Update progress callback
            def update_live_progress():
                live.update(progress_tracker.create_status_panel(), refresh=True)
            
            # Phase 1: Initialize
            progress_tracker.update_phase("Initializing agents...")
            for agent in agents:
                progress_tracker.update_agent_status(agent, 'pending', 'waiting')
            update_live_progress()
            time.sleep(0.5)
            
            # Phase 2: Generation
            progress_tracker.update_phase("Generating collaborative specification...")
            
            # Run generation (async or sync based on orchestrator type)
            try:
                if use_async:
                    # Setup progress callback for async orchestrator
                    def progress_callback(phase: str, details: str, data: dict = None):
                        progress_tracker.update_phase(f"{phase}: {details}")
                        if data and 'agents_completed' in data:
                            for i, agent in enumerate(agents):
                                if i < data['agents_completed']:
                                    progress_tracker.update_agent_status(agent, 'completed', 'done')
                                elif i == data['agents_completed']:
                                    progress_tracker.update_agent_status(agent, 'working', 'processing')
                                else:
                                    progress_tracker.update_agent_status(agent, 'pending', 'waiting')
                        update_live_progress()
                    
                    orchestrator.set_progress_callback(progress_callback)
                    
                    # Run async generation
                    result = await run_async_collaborative_generation(
                        orchestrator, description, agents, budget, complexity
                    )
                else:
                    # Run sync generation
                    result = asyncio.run(
                        run_collaborative_generation(
                            orchestrator, description, agents, budget, 
                            progress_tracker, update_live_progress
                        )
                    )
            except BudgetExceededException as e:
                progress_tracker.update_phase("Budget exceeded")
                console.print(f"\n[red]Budget exceeded: {e}[/red]")
                raise typer.Exit(1)
        
        # Generation complete - show results
        console.print("\n🎉 [bold green]Collaborative Specification Generated![/bold green]\n")
        
        # Show final summary
        budget_manager.show_final_summary(result.token_summary)
        
        # Quality metrics
        quality_table = Table(title="📈 Quality Metrics")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Score", justify="right", style="white")
        quality_table.add_row("Overall Quality", f"{result.quality_score:.1%}")
        quality_table.add_row("Consensus Confidence", f"{result.consensus_result.confidence_score:.1%}")
        quality_table.add_row("Rounds Completed", f"{result.consensus_result.rounds_completed}")
        quality_table.add_row("Unresolved Conflicts", f"{len(result.consensus_result.unresolved_conflicts)}")
        console.print(quality_table)
        
        # Save output
        if output:
            output_path = Path(output)
        else:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"collaborative_spec_{timestamp}.md")
        
        output_path.write_text(result.specification)
        console.print(f"\n📁 Specification saved to: [cyan]{output_path}[/cyan]")
        
        # Show conflicts if any
        if result.consensus_result.unresolved_conflicts:
            console.print("\n⚠️  [yellow]Unresolved conflicts detected:[/yellow]")
            for conflict in result.consensus_result.unresolved_conflicts:
                console.print(f"  • {conflict.description}")
        
        if verbose:
            # Show detailed discussion log
            console.print(f"\n📋 Discussion log saved to: collaborative_spec_{timestamp}_log.json")
            log_path = Path(f"collaborative_spec_{timestamp}_log.json")
            log_path.write_text(json.dumps(result.discussion_log, indent=2))
    
    except Exception as e:
        # Try recovery first
        context = {
            "command": "generate",
            "description": description[:100] + "..." if len(description) > 100 else description,
            "agents": agents,
            "budget": budget,
            "verbose": verbose
        }
        
        # Attempt recovery with the generation function
        async def recovery_operation():
            return await run_collaborative_generation(
                orchestrator, description, agents, budget, 
                progress_tracker, lambda: None
            )
        
        try:
            recovered_result = await recovery_manager.attempt_recovery(
                recovery_operation, e, context
            )
            
            if recovered_result:
                console.print("\n🎉 [bold green]Generation completed via recovery![/bold green]")
                result = recovered_result
                
                # Continue with normal result processing...
                budget_manager.show_final_summary(result.token_summary)
                
                # Save recovered result
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = Path(output) if output else Path(f"collaborative_spec_{timestamp}.md")
                output_path.write_text(result.specification)
                console.print(f"\n📁 Specification saved to: [cyan]{output_path}[/cyan]")
                return
            
        except Exception:
            pass  # Recovery failed, continue with error handling
        
        # Use advanced error handling if recovery fails
        if not error_handler.handle_error(e, context, show_technical=verbose):
            raise typer.Exit(1)
        raise typer.Exit(1)


async def run_collaborative_generation(orchestrator: CollaborativeSpecOrchestrator,
                                     description: str,
                                     agents: List[str], 
                                     budget: float,
                                     progress_tracker: LiveProgressTracker,
                                     update_callback) -> SpecGenerationResult:
    """Run collaborative generation with progress tracking"""
    
    # Custom progress tracking wrapper
    original_gather_responses = orchestrator._gather_initial_responses
    
    async def tracked_gather_responses(agent_instances, user_input):
        progress_tracker.update_phase("Collecting agent responses...")
        
        for agent in agent_instances:
            progress_tracker.update_agent_status(agent.role, 'working', 'analyzing...')
            update_callback()
        
        responses = await original_gather_responses(agent_instances, user_input)
        
        for agent in agent_instances:
            progress_tracker.update_agent_status(agent.role, 'completed', 'analysis done')
        
        progress_tracker.update_token_usage(orchestrator.token_tracker.get_session_summary())
        update_callback()
        
        return responses
    
    # Monkey patch for progress tracking
    orchestrator._gather_initial_responses = tracked_gather_responses
    
    # Run generation
    result = await orchestrator.generate_collaborative_spec(
        user_input=description,
        agent_names=agents,
        budget_limit=budget
    )
    
    progress_tracker.update_phase("Generation complete!")
    progress_tracker.update_token_usage(result.token_summary)
    update_callback()
    
    return result


@app.command("agents")
def list_agents():
    """List available collaborative agents"""
    
    agents_table = Table(title="🤖 Available Collaborative Agents")
    agents_table.add_column("Agent", style="cyan", width=15)
    agents_table.add_column("Role", style="white", width=20)
    agents_table.add_column("Focus Areas", style="dim", width=40)
    agents_table.add_column("Default Model", style="green", width=15)
    
    agent_roles = {
        'pm': 'Product Manager',
        'technical': 'Technical Architect', 
        'security': 'Security Architect',
        'qa': 'QA Engineer'
    }
    
    agent_focus = {
        'pm': 'User value, business requirements, acceptance criteria',
        'technical': 'Architecture, feasibility, implementation',
        'security': 'Threat modeling, compliance, data protection', 
        'qa': 'Test scenarios, quality metrics, edge cases'
    }
    
    for agent_key in AGENT_REGISTRY.keys():
        if agent_key in agent_roles:  # Show main agents only
            config = DEFAULT_AGENT_CONFIGS.get(agent_key, {})
            agents_table.add_row(
                agent_key,
                agent_roles[agent_key],
                agent_focus.get(agent_key, ''),
                config.get('model', 'gpt-4')
            )
    
    console.print(agents_table)
    console.print("\nUsage: specify collab generate 'your description' --agent pm --agent technical")


@app.command("estimate")
def estimate_cost(
    agents: List[str] = typer.Option(["pm", "technical"], "--agent", "-a", help="Agents to include"),
    complexity: str = typer.Option("medium", "--complexity", "-c", help="Complexity level")
):
    """Estimate cost for collaborative generation"""
    
    budget_manager = BudgetManager(console)
    budget_manager.show_cost_estimate(agents, complexity, 999.0)  # High budget to show estimate only


@app.command("version")
def show_version(
    history: bool = typer.Option(False, "--history", "-h", help="Show version history"),
    check_updates: bool = typer.Option(False, "--check-updates", "-u", help="Check for available updates")
):
    """Show plugin version and compatibility information"""
    
    version_manager.show_version_info(include_history=history)
    
    if check_updates:
        console.print("\n🔍 Checking for updates...")
        upgrade_info = version_manager.check_plugin_upgrade()
        if not upgrade_info:
            console.print("✅ [green]No updates available[/green]")


@app.command("migrate")
def migrate_version(
    from_version: str = typer.Argument(..., help="Version to migrate from"),
    to_version: str = typer.Option(None, help="Version to migrate to (current if not specified)")
):
    """Show migration guide between plugin versions"""
    
    if not to_version:
        to_version = version_manager.get_current_version()
    
    # Validate versions
    from_info = version_manager.get_version_info(from_version)
    to_info = version_manager.get_version_info(to_version)
    
    if not from_info:
        console.print(f"[red]Unknown version: {from_version}[/red]")
        console.print("Available versions: " + ", ".join(version_manager.VERSION_HISTORY.keys()))
        raise typer.Exit(1)
    
    if not to_info:
        console.print(f"[red]Unknown version: {to_version}[/red]")
        raise typer.Exit(1)
    
    version_manager.show_migration_guide(from_version, to_version)


@app.command("recovery-stats")
def show_recovery_stats():
    """Show error recovery statistics"""
    recovery_manager.show_recovery_stats()


async def run_async_collaborative_generation(
    orchestrator: AsyncCollaborativeOrchestrator,
    description: str,
    agents: List[str],
    budget: float,
    complexity: str
) -> SpecGenerationResult:
    """Run async collaborative generation"""
    
    try:
        async with orchestrator:  # Use async context manager
            result = await orchestrator.generate_collaborative_spec(
                description=description,
                agents=agents,
                max_budget=budget,
                complexity=complexity,
                enable_streaming=True
            )
            return result
    except Exception as e:
        # Convert to sync exception for compatibility
        raise e


if __name__ == "__main__":
    app()