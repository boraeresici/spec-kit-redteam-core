#!/usr/bin/env python3
"""
Template CLI Commands

Command-line interface for template management and selection.
Provides interactive template browsing, recommendation, and selection.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from pathlib import Path

from ..templates.template_engine import (
    TemplateManager, SecurityTemplate, TemplateCategory, 
    SecurityFramework, TemplateComplexity, TierRequired
)
from ..templates.recommendation_engine import (
    TemplateRecommendationEngine, TemplateRecommendation
)

app = typer.Typer(help="Template management and selection commands")
console = Console()


class TemplateDisplay:
    """Handles rich display formatting for templates."""
    
    @staticmethod
    def display_template_summary(template: SecurityTemplate) -> Panel:
        """Display template summary in a panel."""
        
        # Create content text
        content = Text()
        
        # Basic info
        content.append(f"Category: ", style="bold")
        content.append(f"{template.category.value}\n")
        
        content.append(f"Complexity: ", style="bold")
        complexity_color = {
            "low": "green", 
            "medium": "yellow", 
            "high": "red", 
            "critical": "red bold"
        }.get(template.complexity_level.value, "white")
        content.append(f"{template.complexity_level.value}\n", style=complexity_color)
        
        content.append(f"Estimated Cost: ", style="bold")
        content.append(f"${template.estimated_cost:.2f}\n", style="cyan")
        
        content.append(f"Time: ", style="bold")
        content.append(f"{template.estimated_time_minutes} min\n")
        
        # Agents
        content.append(f"Required Agents: ", style="bold")
        content.append(f"{', '.join(template.required_agents)}\n", style="blue")
        
        # Frameworks
        if template.security_frameworks:
            content.append(f"Frameworks: ", style="bold")
            frameworks = [f.value for f in template.security_frameworks]
            content.append(f"{', '.join(frameworks)}\n", style="green")
        
        # Tier requirement
        if template.tier_required != TierRequired.FREE:
            content.append(f"Requires: ", style="bold")
            content.append(f"{template.tier_required.value.upper()} subscription\n", style="red")
        
        # Create panel
        title = f"[bold]{template.name}[/bold]"
        subtitle = template.description[:80] + "..." if len(template.description) > 80 else template.description
        
        return Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style="blue",
            padding=(1, 2)
        )
    
    @staticmethod
    def display_template_details(template: SecurityTemplate) -> None:
        """Display full template details."""
        console.print(f"\n[bold cyan]Template: {template.name}[/bold cyan]")
        console.print(f"[dim]{template.description}[/dim]")
        console.print()
        
        # Create details table
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="bold", width=20)
        table.add_column("Value")
        
        table.add_row("ID", template.id)
        table.add_row("Category", template.category.value)
        table.add_row("Complexity", template.complexity_level.value)
        table.add_row("Tier Required", template.tier_required.value)
        table.add_row("Estimated Cost", f"${template.estimated_cost:.2f}")
        table.add_row("Estimated Time", f"{template.estimated_time_minutes} minutes")
        
        if template.required_agents:
            table.add_row("Required Agents", ", ".join(template.required_agents))
        
        if template.optional_agents:
            table.add_row("Optional Agents", ", ".join(template.optional_agents))
        
        if template.security_frameworks:
            frameworks = [f.value for f in template.security_frameworks]
            table.add_row("Security Frameworks", ", ".join(frameworks))
        
        console.print(table)
        
        # Show template content preview
        if template.template_content:
            console.print(f"\n[bold]Template Content Preview:[/bold]")
            preview = template.template_content[:500] + "..." if len(template.template_content) > 500 else template.template_content
            console.print(Panel(preview, border_style="dim"))
    
    @staticmethod
    def display_recommendations(recommendations: List[TemplateRecommendation]) -> None:
        """Display template recommendations."""
        if not recommendations:
            console.print("[yellow]No template recommendations found.[/yellow]")
            return
        
        console.print(f"\n[bold cyan]Template Recommendations:[/bold cyan]")
        
        for i, rec in enumerate(recommendations, 1):
            confidence_pct = int(rec.confidence_score * 100)
            
            # Create recommendation panel
            content = Text()
            content.append(f"Confidence: {confidence_pct}%\n", style="green" if confidence_pct > 70 else "yellow")
            content.append(f"Cost: ${rec.estimated_cost:.2f}\n", style="cyan")
            content.append(f"Category: {rec.template.category.value}\n")
            content.append(f"Complexity: {rec.template.complexity_level.value}\n")
            
            # Add reasoning
            if rec.reasoning:
                content.append("\nWhy recommended:\n", style="bold")
                for reason in rec.reasoning[:3]:  # Show top 3 reasons
                    content.append(f"• {reason}\n", style="dim")
            
            panel = Panel(
                content,
                title=f"[bold]{i}. {rec.template.name}[/bold]",
                subtitle=rec.template.description[:60] + "..." if len(rec.template.description) > 60 else rec.template.description,
                border_style="green" if confidence_pct > 70 else "yellow",
                width=80
            )
            
            console.print(panel)


@app.command("list")
def list_templates(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Filter by security framework"),
    complexity: Optional[str] = typer.Option(None, "--complexity", "-x", help="Filter by complexity level"),
    tier: str = typer.Option("free", "--tier", "-t", help="User tier (free, pro, enterprise)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information")
):
    """List available security templates."""
    
    template_manager = TemplateManager()
    
    try:
        templates = template_manager.list_templates(
            category=category,
            framework=framework, 
            complexity=complexity,
            user_tier=tier
        )
        
        if not templates:
            console.print("[yellow]No templates found matching your criteria.[/yellow]")
            return
        
        console.print(f"\n[bold cyan]Available Templates ({len(templates)} found):[/bold cyan]\n")
        
        if detailed:
            for template in templates:
                TemplateDisplay.display_template_details(template)
                console.print("\n" + "─" * 80 + "\n")
        else:
            # Display in grid format
            panels = [TemplateDisplay.display_template_summary(template) for template in templates]
            console.print(Columns(panels, equal=True, expand=True))
        
        # Show statistics
        stats = template_manager.get_template_stats()
        console.print(f"\n[dim]Total templates: {stats['total_templates']} | Average cost: ${stats['average_cost']:.2f}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error listing templates: {e}[/red]")


@app.command("search")
def search_templates(
    query: str = typer.Argument(..., help="Search query"),
    tier: str = typer.Option("free", "--tier", "-t", help="User tier")
):
    """Search templates by name, description, or tags."""
    
    template_manager = TemplateManager()
    
    try:
        templates = template_manager.search_templates(query, user_tier=tier)
        
        if not templates:
            console.print(f"[yellow]No templates found for query: '{query}'[/yellow]")
            return
        
        console.print(f"\n[bold cyan]Search Results for '{query}' ({len(templates)} found):[/bold cyan]\n")
        
        panels = [TemplateDisplay.display_template_summary(template) for template in templates]
        console.print(Columns(panels, equal=True, expand=True))
        
    except Exception as e:
        console.print(f"[red]Error searching templates: {e}[/red]")


@app.command("recommend")
def recommend_templates(
    description: str = typer.Argument(..., help="Project description for recommendation"),
    tier: str = typer.Option("free", "--tier", "-t", help="User tier"),
    max_results: int = typer.Option(3, "--max", "-m", help="Maximum number of recommendations"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive selection mode")
):
    """Get AI-powered template recommendations based on project description."""
    
    template_manager = TemplateManager()
    recommendation_engine = TemplateRecommendationEngine(template_manager)
    
    try:
        console.print(f"\n[bold]Analyzing project description...[/bold]")
        console.print(f"[dim]'{description}'[/dim]\n")
        
        recommendations = recommendation_engine.recommend_templates(
            description, 
            user_tier=tier, 
            max_recommendations=max_results
        )
        
        if not recommendations:
            console.print("[yellow]No suitable templates found for your project description.[/yellow]")
            console.print("[dim]Try using different keywords or check available templates with 'specify collab templates list'[/dim]")
            return
        
        TemplateDisplay.display_recommendations(recommendations)
        
        if interactive and recommendations:
            console.print("\n[bold]Select a template to use:[/bold]")
            
            choices = []
            for i, rec in enumerate(recommendations, 1):
                choices.append(str(i))
            choices.append("0")  # Option to cancel
            
            choice = Prompt.ask(
                "Enter template number (0 to cancel)",
                choices=choices,
                default="1"
            )
            
            if choice != "0":
                selected_template = recommendations[int(choice) - 1].template
                console.print(f"\n[green]Selected template: {selected_template.name}[/green]")
                
                # Show template details
                TemplateDisplay.display_template_details(selected_template)
                
                if Confirm.ask("\nWould you like to generate a specification with this template?"):
                    console.print("[dim]Use: specify collab generate --template {} \"{}\"[/dim]".format(
                        selected_template.id, description
                    ))
        
    except Exception as e:
        console.print(f"[red]Error getting recommendations: {e}[/red]")


@app.command("show")
def show_template(
    template_id: str = typer.Argument(..., help="Template ID to show"),
    tier: str = typer.Option("free", "--tier", "-t", help="User tier")
):
    """Show detailed information about a specific template."""
    
    template_manager = TemplateManager()
    
    try:
        template = template_manager.get_template(template_id, user_tier=tier)
        TemplateDisplay.display_template_details(template)
        
        # Show full content if requested
        if Confirm.ask("\nWould you like to see the full template content?"):
            console.print(f"\n[bold]Full Template Content:[/bold]")
            console.print(Panel(template.template_content, border_style="blue"))
        
    except Exception as e:
        console.print(f"[red]Error showing template: {e}[/red]")


@app.command("stats")
def show_template_stats():
    """Show template system statistics."""
    
    template_manager = TemplateManager()
    
    try:
        stats = template_manager.get_template_stats()
        
        console.print("\n[bold cyan]Template System Statistics:[/bold cyan]\n")
        
        # Main stats table
        main_table = Table(show_header=False)
        main_table.add_column("Metric", style="bold")
        main_table.add_column("Value", style="cyan")
        
        main_table.add_row("Total Templates", str(stats["total_templates"]))
        main_table.add_row("Average Cost", f"${stats['average_cost']:.2f}")
        
        console.print(main_table)
        
        # Category breakdown
        if stats["by_category"]:
            console.print("\n[bold]Templates by Category:[/bold]")
            category_table = Table()
            category_table.add_column("Category", style="bold")
            category_table.add_column("Count", style="cyan")
            
            for category, count in stats["by_category"].items():
                category_table.add_row(category.title(), str(count))
            
            console.print(category_table)
        
        # Tier breakdown
        if stats["by_tier"]:
            console.print("\n[bold]Templates by Subscription Tier:[/bold]")
            tier_table = Table()
            tier_table.add_column("Tier", style="bold")
            tier_table.add_column("Count", style="cyan")
            
            for tier, count in stats["by_tier"].items():
                tier_table.add_row(tier.upper(), str(count))
            
            console.print(tier_table)
        
        # Framework breakdown
        if stats["by_framework"]:
            console.print("\n[bold]Templates by Security Framework:[/bold]")
            framework_table = Table()
            framework_table.add_column("Framework", style="bold")
            framework_table.add_column("Count", style="cyan")
            
            for framework, count in stats["by_framework"].items():
                framework_table.add_row(framework, str(count))
            
            console.print(framework_table)
        
    except Exception as e:
        console.print(f"[red]Error showing stats: {e}[/red]")


@app.command("wizard")
def template_wizard():
    """Interactive template selection wizard."""
    
    console.print(Panel.fit(
        "[bold cyan]🔴 RedTeam Template Selection Wizard[/bold cyan]\n" +
        "Let's find the perfect security template for your project!",
        border_style="blue"
    ))
    
    try:
        # Step 1: Get project description
        console.print("\n[bold]Step 1: Project Description[/bold]")
        description = Prompt.ask(
            "Describe your project in a few sentences",
            default="web application with user authentication"
        )
        
        # Step 2: Get user tier
        console.print("\n[bold]Step 2: Subscription Tier[/bold]")
        tier = Prompt.ask(
            "What's your subscription tier?",
            choices=["free", "pro", "enterprise"],
            default="free"
        )
        
        # Step 3: Get preferences
        console.print("\n[bold]Step 3: Preferences[/bold]")
        max_cost = Prompt.ask(
            "Maximum budget for generation (USD)",
            default="10.00"
        )
        
        try:
            max_cost_float = float(max_cost)
        except ValueError:
            max_cost_float = 10.0
        
        # Step 4: Get recommendations
        console.print(f"\n[bold]Analyzing your project...[/bold]")
        
        template_manager = TemplateManager()
        recommendation_engine = TemplateRecommendationEngine(template_manager)
        
        recommendations = recommendation_engine.recommend_templates(
            description,
            user_tier=tier,
            max_recommendations=5
        )
        
        # Filter by budget
        affordable_recommendations = [
            rec for rec in recommendations 
            if rec.estimated_cost <= max_cost_float
        ]
        
        if not affordable_recommendations:
            console.print(f"[yellow]No templates found within ${max_cost_float:.2f} budget.[/yellow]")
            if recommendations:
                console.print("[dim]Consider increasing your budget or upgrading your subscription.[/dim]")
                TemplateDisplay.display_recommendations(recommendations[:3])
            return
        
        # Step 5: Show recommendations and let user choose
        TemplateDisplay.display_recommendations(affordable_recommendations)
        
        console.print(f"\n[bold]Select your preferred template:[/bold]")
        
        choices = [str(i) for i in range(1, len(affordable_recommendations) + 1)]
        choices.append("0")
        
        selection = Prompt.ask(
            "Enter template number (0 to exit)",
            choices=choices,
            default="1"
        )
        
        if selection == "0":
            console.print("[yellow]Template selection cancelled.[/yellow]")
            return
        
        # Step 6: Show selected template and generate command
        selected_rec = affordable_recommendations[int(selection) - 1]
        selected_template = selected_rec.template
        
        console.print(f"\n[green]✅ Selected: {selected_template.name}[/green]")
        
        # Show the command to run
        generate_command = f'specify collab generate --template {selected_template.id} "{description}"'
        
        console.print(f"\n[bold]Next Steps:[/bold]")
        console.print(Panel(
            f"Run this command to generate your specification:\n\n[cyan]{generate_command}[/cyan]",
            border_style="green"
        ))
        
        if Confirm.ask("Would you like to run this command now?"):
            console.print(f"[dim]Running: {generate_command}[/dim]")
            # Here you would integrate with the main generation command
            console.print("[yellow]Generation integration pending...[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Template wizard cancelled.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error in template wizard: {e}[/red]")


if __name__ == "__main__":
    app()