#!/usr/bin/env python3
"""
Budget management UI components for collaborative AI specification generation.
"""

import time
from typing import Dict, List, Any
from decimal import Decimal

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.text import Text
from rich.align import Align

import typer

from .token_tracker import TokenTracker
from .agents import DEFAULT_AGENT_CONFIGS


class BudgetManager:
    """Manages budget display, confirmation, and live tracking"""
    
    def __init__(self, console: Console):
        self.console = console
        
        # Cost estimates per agent type and complexity
        self.cost_estimates = {
            'pm': {'simple': 0.30, 'medium': 0.60, 'complex': 1.20},
            'technical': {'simple': 0.40, 'medium': 0.80, 'complex': 1.60},
            'security': {'simple': 0.35, 'medium': 0.70, 'complex': 1.40},
            'qa': {'simple': 0.20, 'medium': 0.40, 'complex': 0.80}
        }
        
        self.consensus_cost = {'simple': 0.50, 'medium': 1.00, 'complex': 2.00}
        
        # Cache hit rates for optimization estimates
        self.cache_savings = {'simple': 0.15, 'medium': 0.25, 'complex': 0.35}
    
    def show_cost_estimate(self, 
                          agents: List[str],
                          complexity: str = 'medium',
                          budget: float = 50.0,
                          enable_cache: bool = True) -> bool:
        """Show detailed cost estimate and get user confirmation"""
        
        self.console.print(f"\n💰 [bold cyan]Collaborative Generation - Cost Estimate[/bold cyan]")
        self.console.print(f"Complexity: [yellow]{complexity.title()}[/yellow] | Budget: [green]${budget:.2f}[/green]")
        self.console.print()
        
        # Main cost table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Agent Model", style="dim", width=15) 
        table.add_column("Est. Tokens", justify="right", width=12)
        table.add_column("Est. Cost", justify="right", style="green", width=12)
        table.add_column("Notes", style="dim", width=25)
        
        total_estimated = 0.0
        total_tokens = 0
        
        # Agent costs
        for agent in agents:
            agent_cost = self.cost_estimates.get(agent, self.cost_estimates['pm'])[complexity]
            agent_tokens = self._estimate_tokens_for_agent(agent, complexity)
            total_estimated += agent_cost
            total_tokens += agent_tokens
            
            # Get model info
            model = DEFAULT_AGENT_CONFIGS.get(agent, {}).get('model', 'gpt-4')
            agent_name = agent.title() + " Agent"
            
            table.add_row(
                agent_name,
                model,
                f"{agent_tokens:,}",
                f"${agent_cost:.2f}",
                f"Analysis + review"
            )
        
        # Consensus costs
        consensus_estimated = self.consensus_cost[complexity]
        consensus_tokens = self._estimate_consensus_tokens(len(agents), complexity)
        total_estimated += consensus_estimated
        total_tokens += consensus_tokens
        
        table.add_row(
            "Consensus Building",
            "Multi-agent",
            f"{consensus_tokens:,}",
            f"${consensus_estimated:.2f}",
            f"{len(agents)}-agent discussion"
        )
        
        # Separator
        table.add_row("", "", "", "", "", style="dim")
        
        # Subtotal
        table.add_row(
            "Subtotal", 
            "", 
            f"{total_tokens:,}",
            f"${total_estimated:.2f}",
            "Before optimizations",
            style="bold"
        )
        
        # Optimizations
        optimized_cost = total_estimated
        if enable_cache:
            cache_savings_amount = total_estimated * self.cache_savings[complexity]
            optimized_cost -= cache_savings_amount
            table.add_row(
                "Cache Optimization",
                "",
                f"-{int(total_tokens * self.cache_savings[complexity]):,}",
                f"-${cache_savings_amount:.2f}",
                f"~{self.cache_savings[complexity]*100:.0f}% token savings",
                style="green"
            )
        
        # Final total
        table.add_row(
            "Total Estimated",
            "",
            f"{int(total_tokens * (1-self.cache_savings[complexity]) if enable_cache else total_tokens):,}",
            f"${optimized_cost:.2f}",
            f"Budget: ${budget:.2f}",
            style="bold green"
        )
        
        self.console.print(table)
        self.console.print()
        
        # Budget analysis
        if optimized_cost > budget:
            self.console.print(f"⚠️  [yellow]Estimated cost (${optimized_cost:.2f}) exceeds budget (${budget:.2f})[/yellow]")
            
            # Show recommendations
            recommendations = []
            if len(agents) > 2:
                recommendations.append("Consider reducing agents (use --agent pm --agent technical for minimum)")
            if complexity == 'complex':
                recommendations.append("Try --complexity medium to reduce costs")
            if not enable_cache:
                recommendations.append("Enable caching (--cache) for savings")
            
            if recommendations:
                self.console.print("[dim]Recommendations:[/dim]")
                for rec in recommendations:
                    self.console.print(f"  • {rec}")
                self.console.print()
            
            return typer.confirm("Proceed anyway?")
        
        elif optimized_cost > budget * 0.8:
            self.console.print(f"⚠️  [yellow]Cost will use {optimized_cost/budget*100:.0f}% of budget[/yellow]")
        
        return typer.confirm(f"Proceed with estimated cost of ${optimized_cost:.2f}?")
    
    def _estimate_tokens_for_agent(self, agent: str, complexity: str) -> int:
        """Estimate token usage for specific agent"""
        base_tokens = {
            'pm': {'simple': 1800, 'medium': 3000, 'complex': 5000},
            'technical': {'simple': 2500, 'medium': 4000, 'complex': 6500},
            'security': {'simple': 2200, 'medium': 3500, 'complex': 5500},
            'qa': {'simple': 1500, 'medium': 2500, 'complex': 4000}
        }
        return base_tokens.get(agent, base_tokens['pm'])[complexity]
    
    def _estimate_consensus_tokens(self, agent_count: int, complexity: str) -> int:
        """Estimate token usage for consensus building"""
        base_consensus = {'simple': 2000, 'medium': 4000, 'complex': 7000}
        return base_consensus[complexity] + (agent_count - 2) * 1000  # Additional cost per extra agent
    
    def show_live_usage(self, token_tracker: TokenTracker):
        """Display live token usage during generation"""
        
        progress_table = Table(title="🔄 Generation in Progress", show_header=True)
        progress_table.add_column("Agent", style="cyan", width=15)
        progress_table.add_column("Status", width=12)
        progress_table.add_column("Tokens", justify="right", width=10)
        progress_table.add_column("Cost", justify="right", style="green", width=8)
        progress_table.add_column("Activity", style="dim", width=20)
        
        return progress_table
    
    def update_live_usage(self, table: Table, token_summary: Dict, agent_statuses: Dict = None):
        """Update live usage table with current data"""
        # Clear existing rows
        table.rows.clear()
        
        agent_breakdown = token_summary.get('agent_breakdown', {})
        
        for agent_name, usage in agent_breakdown.items():
            status = "✅ Complete"
            activity = "Analysis done"
            
            if agent_statuses and agent_name in agent_statuses:
                agent_status = agent_statuses[agent_name]
                if agent_status['status'] == 'working':
                    status = "🔄 Working"
                    activity = agent_status.get('details', 'Processing...')
                elif agent_status['status'] == 'pending':
                    status = "⏳ Pending" 
                    activity = "Waiting..."
                elif agent_status['status'] == 'error':
                    status = "❌ Error"
                    activity = agent_status.get('details', 'Failed')
            
            table.add_row(
                agent_name,
                status,
                f"{usage.get('total_tokens', 0):,}",
                f"${usage.get('total_cost', 0):.2f}",
                activity
            )
        
        # Add summary row
        table.add_row("", "", "", "", "", style="dim")
        table.add_row(
            "Total",
            "",
            f"{token_summary.get('total_tokens', 0):,}",
            f"${token_summary.get('total_cost', 0):.2f}",
            f"Budget: {token_summary.get('budget_used_percentage', 0):.0f}% used",
            style="bold"
        )
    
    def show_final_summary(self, token_summary: Dict, generation_metadata: Dict = None):
        """Show comprehensive final summary"""
        
        self.console.print("\n📊 [bold green]Generation Complete - Final Summary[/bold green]")
        
        # Main metrics table
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", justify="right", style="white", width=15)
        summary_table.add_column("Details", style="dim", width=30)
        
        # Basic metrics
        duration = token_summary.get('duration_seconds', 0)
        summary_table.add_row(
            "Generation Time", 
            f"{duration:,}s",
            f"{duration/60:.1f} minutes" if duration > 60 else "Fast generation"
        )
        
        summary_table.add_row(
            "Total Tokens", 
            f"{token_summary.get('total_tokens', 0):,}",
            f"{token_summary.get('operation_count', 0)} operations"
        )
        
        summary_table.add_row(
            "Total Cost", 
            f"${token_summary.get('total_cost', 0):.2f}",
            "All agents + consensus"
        )
        
        # Budget utilization
        if token_summary.get('budget_limit'):
            usage_pct = token_summary.get('budget_used_percentage', 0)
            budget_status = "Within budget" if usage_pct <= 100 else "Over budget"
            summary_table.add_row(
                "Budget Used",
                f"{usage_pct:.1f}%",
                budget_status
            )
        
        # Agent efficiency
        agent_count = len(token_summary.get('agent_breakdown', {}))
        avg_cost_per_agent = token_summary.get('total_cost', 0) / max(1, agent_count)
        summary_table.add_row(
            "Agents Used",
            f"{agent_count}",
            f"${avg_cost_per_agent:.2f} avg/agent"
        )
        
        self.console.print(summary_table)
        
        # Agent breakdown table
        if token_summary.get('agent_breakdown'):
            self.console.print("\n🤖 [bold]Agent Performance Breakdown[/bold]")
            
            agent_table = Table(show_header=True, header_style="bold cyan")
            agent_table.add_column("Agent", style="cyan")
            agent_table.add_column("Tokens", justify="right")
            agent_table.add_column("Cost", justify="right", style="green")
            agent_table.add_column("Efficiency", justify="right")
            agent_table.add_column("Model", style="dim")
            
            for agent_name, usage in token_summary['agent_breakdown'].items():
                efficiency = usage.get('total_tokens', 0) / max(1, usage.get('operations', 1))
                model = usage.get('model', 'Unknown')
                
                agent_table.add_row(
                    agent_name,
                    f"{usage.get('total_tokens', 0):,}",
                    f"${usage.get('total_cost', 0):.2f}",
                    f"{efficiency:.0f} tok/op",
                    model
                )
            
            self.console.print(agent_table)
        
        # Optimization summary
        if generation_metadata:
            optimization_info = []
            
            if generation_metadata.get('cache_hits'):
                cache_efficiency = generation_metadata.get('cache_efficiency', 0)
                optimization_info.append(f"Cache: {cache_efficiency:.0f}% hit rate")
            
            if generation_metadata.get('parallel_execution'):
                optimization_info.append("Parallel: Enabled")
            
            budget_util = generation_metadata.get('budget_utilization', 0)
            if budget_util < 80:
                optimization_info.append(f"Budget: Efficient ({budget_util:.0f}% used)")
            
            if optimization_info:
                self.console.print(f"\n✨ [green]Optimizations:[/green] {' • '.join(optimization_info)}")
    
    def show_budget_warning(self, current_cost: float, budget_limit: float, remaining_operations: int = 0):
        """Show budget warning when approaching limits"""
        
        usage_pct = (current_cost / budget_limit) * 100
        
        if usage_pct >= 90:
            self.console.print(f"\n🚨 [red]Budget Alert:[/red] {usage_pct:.0f}% used (${current_cost:.2f}/${budget_limit:.2f})")
            
            if remaining_operations > 0:
                est_final_cost = current_cost * (1 + remaining_operations * 0.3)  # Rough estimate
                if est_final_cost > budget_limit:
                    self.console.print(f"⚠️  [yellow]Projected final cost: ${est_final_cost:.2f}[/yellow]")
                    return typer.confirm("Continue generation despite potential budget overrun?")
        
        elif usage_pct >= 70:
            self.console.print(f"\n💡 [yellow]Budget Notice:[/yellow] {usage_pct:.0f}% used")
        
        return True
    
    def show_cost_comparison(self, collaborative_cost: float, traditional_estimates: Dict = None):
        """Show cost comparison with traditional development approaches"""
        
        if not traditional_estimates:
            traditional_estimates = {
                'manual_spec_writing': 200.0,  # Developer time @ $100/hr for 2 hours
                'review_meetings': 300.0,       # 3 people @ $100/hr for 1 hour
                'revision_cycles': 150.0,       # Additional revisions
                'total_traditional': 650.0
            }
        
        self.console.print("\n💰 [bold]Cost Comparison[/bold]")
        
        comparison_table = Table(show_header=True, header_style="bold cyan")
        comparison_table.add_column("Approach", style="cyan")
        comparison_table.add_column("Cost", justify="right", style="green")
        comparison_table.add_column("Time", justify="right") 
        comparison_table.add_column("Quality", justify="center")
        
        comparison_table.add_row(
            "Collaborative AI",
            f"${collaborative_cost:.2f}",
            "~5 minutes",
            "🌟🌟🌟🌟🌟"
        )
        
        comparison_table.add_row(
            "Traditional Process",
            f"${traditional_estimates['total_traditional']:.2f}",
            "~4 hours",
            "🌟🌟🌟"
        )
        
        savings = traditional_estimates['total_traditional'] - collaborative_cost
        time_savings = "95% faster"
        
        comparison_table.add_row("", "", "", "", style="dim")
        comparison_table.add_row(
            "Savings",
            f"${savings:.2f}",
            time_savings,
            "Higher quality",
            style="bold green"
        )
        
        self.console.print(comparison_table)
        
        if savings > 0:
            self.console.print(f"\n💡 [green]You're saving ${savings:.2f} and getting {time_savings} delivery![/green]")