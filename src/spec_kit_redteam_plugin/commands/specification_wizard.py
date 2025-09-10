#!/usr/bin/env python3
"""
Interactive Specification Wizard

Multi-step wizard interface for guided specification generation.
Provides project type selection, security requirements questionnaire,
and configuration persistence with progress saving.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.columns import Columns
from rich.layout import Layout
from rich.align import Align

from ..templates.template_engine import (
    TemplateManager, SecurityTemplate, TemplateCategory, 
    SecurityFramework, TemplateComplexity, TierRequired
)
from ..templates.recommendation_engine import (
    TemplateRecommendationEngine, TemplateRecommendation, ProjectType, DataSensitivity
)


class WizardStep(Enum):
    """Wizard step enumeration"""
    WELCOME = "welcome"
    PROJECT_TYPE = "project_type"
    PROJECT_DETAILS = "project_details"  
    SECURITY_REQUIREMENTS = "security_requirements"
    AGENT_SELECTION = "agent_selection"
    BUDGET_CONFIGURATION = "budget_configuration"
    TEMPLATE_RECOMMENDATION = "template_recommendation"
    FINAL_CONFIRMATION = "final_confirmation"
    COMPLETED = "completed"


@dataclass
class ProjectConfiguration:
    """Project configuration collected through wizard"""
    
    # Basic project info
    project_name: str = ""
    project_description: str = ""
    project_type: str = ""
    
    # Project details
    technologies: List[str] = field(default_factory=list)
    deployment_environment: str = ""
    expected_users: int = 0
    
    # Security requirements
    data_sensitivity: str = "internal"
    compliance_frameworks: List[str] = field(default_factory=list)
    security_priorities: List[str] = field(default_factory=list)
    
    # Agent configuration
    selected_agents: List[str] = field(default_factory=list)
    agent_priorities: Dict[str, int] = field(default_factory=dict)
    
    # Budget and constraints
    max_budget: float = 10.0
    time_constraint: int = 60  # minutes
    quality_preference: str = "balanced"  # speed, balanced, quality
    
    # Template selection
    recommended_templates: List[Dict] = field(default_factory=list)
    selected_template: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    wizard_version: str = "1.0.0"
    completion_status: str = "in_progress"


@dataclass 
class WizardSession:
    """Wizard session state management"""
    session_id: str
    current_step: WizardStep = WizardStep.WELCOME
    configuration: ProjectConfiguration = field(default_factory=ProjectConfiguration)
    step_history: List[WizardStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_saved_at: Optional[datetime] = None
    
    def save_to_file(self, file_path: Path):
        """Save session to file"""
        session_data = {
            'session_id': self.session_id,
            'current_step': self.current_step.value,
            'configuration': asdict(self.configuration),
            'step_history': [step.value for step in self.step_history],
            'started_at': self.started_at.isoformat(),
            'last_saved_at': datetime.utcnow().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self.last_saved_at = datetime.utcnow()
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'WizardSession':
        """Load session from file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        session = cls(session_id=data['session_id'])
        session.current_step = WizardStep(data['current_step'])
        session.configuration = ProjectConfiguration(**data['configuration'])
        session.step_history = [WizardStep(step) for step in data['step_history']]
        session.started_at = datetime.fromisoformat(data['started_at'])
        if data.get('last_saved_at'):
            session.last_saved_at = datetime.fromisoformat(data['last_saved_at'])
        
        return session


class SpecificationWizard:
    """Interactive multi-step specification generation wizard"""
    
    def __init__(self):
        self.console = Console()
        self.template_manager = TemplateManager()
        self.recommendation_engine = TemplateRecommendationEngine(self.template_manager)
        self.sessions_dir = Path.home() / ".spec-kit" / "wizard_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Wizard configuration
        self.allow_back_navigation = True
        self.auto_save = True
        self.save_interval = 300  # seconds
        
    def start_wizard(self, resume_session_id: Optional[str] = None) -> ProjectConfiguration:
        """Start interactive specification wizard"""
        
        try:
            # Initialize or resume session
            if resume_session_id:
                session = self._resume_session(resume_session_id)
            else:
                session = self._create_new_session()
            
            # Main wizard loop
            while session.current_step != WizardStep.COMPLETED:
                # Auto-save periodically
                if self.auto_save and self._should_auto_save(session):
                    self._save_session(session)
                
                # Execute current step
                success = self._execute_step(session)
                
                if success:
                    # Move to next step
                    session = self._advance_step(session)
                else:
                    # Handle step failure or back navigation
                    if self.allow_back_navigation and len(session.step_history) > 0:
                        session = self._go_back_step(session)
                    else:
                        self.console.print("[red]Unable to complete wizard step. Exiting.[/red]")
                        break
            
            # Final save and cleanup
            session.configuration.completion_status = "completed"
            self._save_session(session)
            
            return session.configuration
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Wizard interrupted. Progress saved.[/yellow]")
            if 'session' in locals():
                self._save_session(session)
            return None
        except Exception as e:
            self.console.print(f"[red]Wizard error: {e}[/red]")
            return None
    
    def _create_new_session(self) -> WizardSession:
        """Create new wizard session"""
        session_id = f"wizard_{int(time.time())}"
        session = WizardSession(session_id=session_id)
        
        self.console.print(Panel.fit(
            "[bold cyan]🧙‍♂️ RedTeam Specification Wizard[/bold cyan]\n" +
            "Let's create a comprehensive security specification for your project!\n\n" +
            f"Session ID: [dim]{session_id}[/dim]",
            border_style="blue",
            title="Welcome"
        ))
        
        return session
    
    def _resume_session(self, session_id: str) -> WizardSession:
        """Resume existing wizard session"""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            raise ValueError(f"Session {session_id} not found")
        
        session = WizardSession.load_from_file(session_file)
        
        self.console.print(Panel.fit(
            f"[bold green]🔄 Resuming Wizard Session[/bold green]\n" +
            f"Session: {session_id}\n" +
            f"Current Step: {session.current_step.value.replace('_', ' ').title()}\n" +
            f"Started: {session.started_at.strftime('%Y-%m-%d %H:%M')}",
            border_style="green",
            title="Session Resumed"
        ))
        
        return session
    
    def _execute_step(self, session: WizardSession) -> bool:
        """Execute current wizard step"""
        
        step_methods = {
            WizardStep.WELCOME: self._step_welcome,
            WizardStep.PROJECT_TYPE: self._step_project_type,
            WizardStep.PROJECT_DETAILS: self._step_project_details,
            WizardStep.SECURITY_REQUIREMENTS: self._step_security_requirements,
            WizardStep.AGENT_SELECTION: self._step_agent_selection,
            WizardStep.BUDGET_CONFIGURATION: self._step_budget_configuration,
            WizardStep.TEMPLATE_RECOMMENDATION: self._step_template_recommendation,
            WizardStep.FINAL_CONFIRMATION: self._step_final_confirmation
        }
        
        step_method = step_methods.get(session.current_step)
        if step_method:
            return step_method(session)
        else:
            self.console.print(f"[red]Unknown wizard step: {session.current_step}[/red]")
            return False
    
    def _step_welcome(self, session: WizardSession) -> bool:
        """Welcome step - introduction and basic info"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 1: Welcome & Project Overview[/bold cyan]")
        self.console.print("="*60)
        
        # Project name
        project_name = Prompt.ask(
            "\n[bold]What's the name of your project?[/bold]",
            default=session.configuration.project_name or "My Secure Application"
        )
        session.configuration.project_name = project_name
        
        # Project description
        description = Prompt.ask(
            "\n[bold]Describe your project in a few sentences[/bold]",
            default=session.configuration.project_description or ""
        )
        session.configuration.project_description = description
        
        # Quick overview
        self.console.print(f"\n[green]✅ Project Overview Set:[/green]")
        self.console.print(f"   • Name: {project_name}")
        self.console.print(f"   • Description: {description[:100]}{'...' if len(description) > 100 else ''}")
        
        return Confirm.ask("\n[bold]Continue to project type selection?[/bold]", default=True)
    
    def _step_project_type(self, session: WizardSession) -> bool:
        """Project type selection step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 2: Project Type Selection[/bold cyan]")
        self.console.print("="*60)
        
        # Project type options
        project_types = {
            "1": ("web", "Web Application", "Traditional web app with frontend/backend"),
            "2": ("api", "REST API", "RESTful API service or microservice"),  
            "3": ("mobile", "Mobile Application", "iOS/Android mobile application"),
            "4": ("microservice", "Microservice", "Containerized microservice architecture"),
            "5": ("desktop", "Desktop Application", "Desktop GUI application"),
            "6": ("infrastructure", "Infrastructure", "Cloud infrastructure or DevOps"),
            "7": ("iot", "IoT Device", "Internet of Things device or system")
        }
        
        # Display options
        table = Table(title="🏗️  Available Project Types")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Type", style="bold")
        table.add_column("Description", style="dim")
        
        for key, (_, name, desc) in project_types.items():
            table.add_row(key, name, desc)
        
        self.console.print(table)
        
        # Get selection
        choice = Prompt.ask(
            "\n[bold]Select your project type[/bold]",
            choices=list(project_types.keys()),
            default="1"
        )
        
        project_type, type_name, type_desc = project_types[choice]
        session.configuration.project_type = project_type
        
        self.console.print(f"\n[green]✅ Selected:[/green] {type_name} - {type_desc}")
        
        return True
    
    def _step_project_details(self, session: WizardSession) -> bool:
        """Project details collection step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 3: Project Details[/bold cyan]")
        self.console.print("="*60)
        
        # Technologies
        self.console.print("\n[bold]🛠️  Technologies & Frameworks[/bold]")
        self.console.print("[dim]Enter technologies used in your project (comma-separated)[/dim]")
        
        tech_input = Prompt.ask(
            "Technologies",
            default=", ".join(session.configuration.technologies) if session.configuration.technologies else "Python, PostgreSQL, Docker"
        )
        session.configuration.technologies = [tech.strip() for tech in tech_input.split(",")]
        
        # Deployment environment
        deployment_options = {
            "1": "cloud",
            "2": "on-premise", 
            "3": "hybrid",
            "4": "edge"
        }
        
        self.console.print("\n[bold]☁️  Deployment Environment[/bold]")
        deploy_choice = Prompt.ask(
            "Where will this be deployed? (1=Cloud, 2=On-Premise, 3=Hybrid, 4=Edge)",
            choices=list(deployment_options.keys()),
            default="1"
        )
        session.configuration.deployment_environment = deployment_options[deploy_choice]
        
        # Expected users
        expected_users = IntPrompt.ask(
            "\n[bold]👥 Expected number of users[/bold]",
            default=session.configuration.expected_users or 1000
        )
        session.configuration.expected_users = expected_users
        
        # Show summary
        self.console.print(f"\n[green]✅ Project Details:[/green]")
        self.console.print(f"   • Technologies: {', '.join(session.configuration.technologies)}")
        self.console.print(f"   • Deployment: {session.configuration.deployment_environment}")
        self.console.print(f"   • Expected Users: {session.configuration.expected_users:,}")
        
        return True
    
    def _step_security_requirements(self, session: WizardSession) -> bool:
        """Security requirements questionnaire step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 4: Security Requirements[/bold cyan]")
        self.console.print("="*60)
        
        # Data sensitivity
        sensitivity_options = {
            "1": ("public", "Public", "Publicly accessible information"),
            "2": ("internal", "Internal", "Internal company information"),
            "3": ("confidential", "Confidential", "Sensitive business information"),
            "4": ("restricted", "Restricted", "Highly classified information")
        }
        
        self.console.print("\n[bold]🔒 Data Sensitivity Level[/bold]")
        sens_table = Table()
        sens_table.add_column("Option", style="cyan")
        sens_table.add_column("Level", style="bold")
        sens_table.add_column("Description", style="dim")
        
        for key, (_, level, desc) in sensitivity_options.items():
            sens_table.add_row(key, level, desc)
        
        self.console.print(sens_table)
        
        sens_choice = Prompt.ask(
            "Select data sensitivity level",
            choices=list(sensitivity_options.keys()),
            default="2"
        )
        session.configuration.data_sensitivity = sensitivity_options[sens_choice][0]
        
        # Compliance frameworks
        self.console.print("\n[bold]📋 Compliance Frameworks[/bold]")
        compliance_options = [
            "OWASP", "NIST", "ISO27001", "SOC2", "GDPR", "HIPAA", "PCI-DSS"
        ]
        
        selected_compliance = []
        for framework in compliance_options:
            if Confirm.ask(f"   Does your project need to comply with {framework}?", default=False):
                selected_compliance.append(framework)
        
        session.configuration.compliance_frameworks = selected_compliance
        
        # Security priorities
        self.console.print("\n[bold]🎯 Security Priorities[/bold]")
        priority_options = [
            "authentication", "authorization", "data_encryption", "input_validation",
            "secure_communication", "audit_logging", "access_control", "vulnerability_management"
        ]
        
        selected_priorities = []
        for priority in priority_options:
            if Confirm.ask(f"   Is {priority.replace('_', ' ')} a high priority?", default=False):
                selected_priorities.append(priority)
        
        session.configuration.security_priorities = selected_priorities
        
        # Show summary
        self.console.print(f"\n[green]✅ Security Requirements:[/green]")
        self.console.print(f"   • Data Sensitivity: {session.configuration.data_sensitivity}")
        self.console.print(f"   • Compliance: {', '.join(session.configuration.compliance_frameworks) or 'None specified'}")
        self.console.print(f"   • Priorities: {', '.join(session.configuration.security_priorities) or 'None specified'}")
        
        return True
    
    def _step_agent_selection(self, session: WizardSession) -> bool:
        """Agent selection and prioritization step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 5: Agent Selection[/bold cyan]")
        self.console.print("="*60)
        
        # Available agents
        agent_descriptions = {
            "pm": ("Project Manager", "Requirements analysis and project coordination"),
            "technical": ("Technical Architect", "Technical design and implementation planning"),
            "security": ("Security Specialist", "Security analysis and threat modeling"),
            "qa": ("Quality Assurance", "Testing strategy and quality validation"),
            "devops": ("DevOps Engineer", "Deployment and infrastructure considerations"),
            "compliance": ("Compliance Officer", "Regulatory compliance and audit requirements")
        }
        
        # Display agent options
        agent_table = Table(title="🤖 Available AI Agents")
        agent_table.add_column("Agent", style="bold")
        agent_table.add_column("Role", style="cyan")
        agent_table.add_column("Description", style="dim")
        agent_table.add_column("Select?", style="green")
        
        selected_agents = []
        
        for agent_id, (role, description) in agent_descriptions.items():
            # Default selections based on project type
            default_agents = {
                "web": ["pm", "technical", "security", "qa"],
                "api": ["technical", "security", "qa"],
                "mobile": ["pm", "technical", "security"],
                "microservice": ["technical", "security", "devops"],
                "infrastructure": ["technical", "devops", "security"]
            }
            
            is_default = agent_id in default_agents.get(session.configuration.project_type, [])
            
            if Confirm.ask(f"Include {role} ({agent_id})?", default=is_default):
                selected_agents.append(agent_id)
                agent_table.add_row(agent_id, role, description, "✅")
            else:
                agent_table.add_row(agent_id, role, description, "❌")
        
        self.console.print(agent_table)
        
        if not selected_agents:
            self.console.print("[yellow]⚠️  No agents selected. Adding recommended agents.[/yellow]")
            selected_agents = ["pm", "technical", "security"]
        
        session.configuration.selected_agents = selected_agents
        
        # Agent prioritization
        if len(selected_agents) > 1:
            self.console.print("\n[bold]📊 Agent Prioritization[/bold]")
            self.console.print("[dim]Rank agents by importance (1 = highest priority)[/dim]")
            
            priorities = {}
            for i, agent in enumerate(selected_agents, 1):
                role = agent_descriptions[agent][0]
                priority = IntPrompt.ask(
                    f"Priority for {role} ({agent})",
                    default=i
                )
                priorities[agent] = priority
            
            session.configuration.agent_priorities = priorities
        
        self.console.print(f"\n[green]✅ Selected Agents:[/green] {', '.join(selected_agents)}")
        
        return True
    
    def _step_budget_configuration(self, session: WizardSession) -> bool:
        """Budget and constraints configuration step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 6: Budget & Constraints[/bold cyan]")
        self.console.print("="*60)
        
        # Budget configuration
        max_budget = FloatPrompt.ask(
            "\n[bold]💰 Maximum budget (USD)[/bold]",
            default=session.configuration.max_budget
        )
        session.configuration.max_budget = max_budget
        
        # Time constraint
        time_constraint = IntPrompt.ask(
            "\n[bold]⏱️  Time constraint (minutes)[/bold]",
            default=session.configuration.time_constraint
        )
        session.configuration.time_constraint = time_constraint
        
        # Quality preference
        quality_options = {
            "1": ("speed", "Speed", "Fast generation, basic quality"),
            "2": ("balanced", "Balanced", "Balance between speed and quality"), 
            "3": ("quality", "Quality", "High quality, slower generation")
        }
        
        self.console.print("\n[bold]⚖️  Quality Preference[/bold]")
        qual_table = Table()
        qual_table.add_column("Option", style="cyan")
        qual_table.add_column("Setting", style="bold")
        qual_table.add_column("Description", style="dim")
        
        for key, (setting, name, desc) in quality_options.items():
            qual_table.add_row(key, name, desc)
        
        self.console.print(qual_table)
        
        qual_choice = Prompt.ask(
            "Select quality preference",
            choices=list(quality_options.keys()),
            default="2"
        )
        session.configuration.quality_preference = quality_options[qual_choice][0]
        
        # Show cost estimate
        estimated_cost = self._estimate_generation_cost(session.configuration)
        
        self.console.print(f"\n[green]✅ Budget Configuration:[/green]")
        self.console.print(f"   • Max Budget: ${session.configuration.max_budget:.2f}")
        self.console.print(f"   • Time Limit: {session.configuration.time_constraint} minutes")
        self.console.print(f"   • Quality: {session.configuration.quality_preference}")
        self.console.print(f"   • Estimated Cost: ${estimated_cost:.2f}")
        
        if estimated_cost > max_budget:
            self.console.print(f"[yellow]⚠️  Estimated cost exceeds budget. Consider reducing agents or complexity.[/yellow]")
            return Confirm.ask("Continue anyway?", default=True)
        
        return True
    
    def _step_template_recommendation(self, session: WizardSession) -> bool:
        """Template recommendation and selection step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 7: Template Recommendation[/bold cyan]")
        self.console.print("="*60)
        
        # Generate recommendations based on collected info
        self.console.print("[dim]🔍 Analyzing your requirements and finding suitable templates...[/dim]")
        
        recommendations = self.recommendation_engine.recommend_templates(
            description=session.configuration.project_description,
            user_tier="free",  # Could be configurable
            max_recommendations=5
        )
        
        if not recommendations:
            self.console.print("[yellow]⚠️  No template recommendations found. You can proceed without a template.[/yellow]")
            session.configuration.selected_template = None
            return True
        
        # Display recommendations
        self.console.print(f"\n[bold]🎯 Found {len(recommendations)} Recommended Templates[/bold]")
        
        for i, rec in enumerate(recommendations, 1):
            confidence_pct = int(rec.confidence_score * 100)
            
            panel_content = Text()
            panel_content.append(f"Confidence: {confidence_pct}%\n", style="green" if confidence_pct > 70 else "yellow")
            panel_content.append(f"Cost: ${rec.estimated_cost:.2f}\n", style="cyan")
            panel_content.append(f"Category: {rec.template.category.value}\n")
            panel_content.append(f"Agents: {', '.join(rec.template.required_agents)}\n", style="blue")
            
            if rec.reasoning:
                panel_content.append("\nWhy recommended:\n", style="bold")
                for reason in rec.reasoning[:2]:
                    panel_content.append(f"• {reason}\n", style="dim")
            
            panel = Panel(
                panel_content,
                title=f"[bold]{i}. {rec.template.name}[/bold]",
                border_style="green" if confidence_pct > 70 else "yellow"
            )
            self.console.print(panel)
        
        # Template selection
        choices = [str(i) for i in range(1, len(recommendations) + 1)] + ["0"]
        choice = Prompt.ask(
            "\n[bold]Select a template (0 for no template)[/bold]",
            choices=choices,
            default="1"
        )
        
        if choice == "0":
            session.configuration.selected_template = None
            self.console.print("[dim]Proceeding without template[/dim]")
        else:
            selected_rec = recommendations[int(choice) - 1]
            session.configuration.selected_template = selected_rec.template.id
            
            # Store recommendation details
            session.configuration.recommended_templates = [
                {
                    "template_id": rec.template.id,
                    "template_name": rec.template.name,
                    "confidence_score": rec.confidence_score,
                    "estimated_cost": rec.estimated_cost
                }
                for rec in recommendations
            ]
            
            self.console.print(f"[green]✅ Selected:[/green] {selected_rec.template.name}")
        
        return True
    
    def _step_final_confirmation(self, session: WizardSession) -> bool:
        """Final confirmation and summary step"""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]Step 8: Final Confirmation[/bold cyan]")
        self.console.print("="*60)
        
        # Configuration summary
        config = session.configuration
        
        summary_panel = Text()
        summary_panel.append("📋 Project Configuration Summary\n\n", style="bold cyan")
        
        summary_panel.append("🏗️  Project Information:\n", style="bold")
        summary_panel.append(f"   • Name: {config.project_name}\n")
        summary_panel.append(f"   • Type: {config.project_type}\n") 
        summary_panel.append(f"   • Technologies: {', '.join(config.technologies[:3])}{'...' if len(config.technologies) > 3 else ''}\n")
        summary_panel.append(f"   • Users: {config.expected_users:,}\n\n")
        
        summary_panel.append("🔒 Security Configuration:\n", style="bold")
        summary_panel.append(f"   • Data Sensitivity: {config.data_sensitivity}\n")
        summary_panel.append(f"   • Compliance: {', '.join(config.compliance_frameworks) or 'None'}\n")
        summary_panel.append(f"   • Security Priorities: {len(config.security_priorities)} selected\n\n")
        
        summary_panel.append("🤖 Agent Configuration:\n", style="bold")
        summary_panel.append(f"   • Selected Agents: {', '.join(config.selected_agents)}\n")
        summary_panel.append(f"   • Agent Count: {len(config.selected_agents)}\n\n")
        
        summary_panel.append("💰 Budget & Constraints:\n", style="bold")
        summary_panel.append(f"   • Max Budget: ${config.max_budget:.2f}\n")
        summary_panel.append(f"   • Time Limit: {config.time_constraint} minutes\n")
        summary_panel.append(f"   • Quality: {config.quality_preference}\n\n")
        
        if config.selected_template:
            summary_panel.append("📄 Template Selection:\n", style="bold")
            template_name = next(
                (t["template_name"] for t in config.recommended_templates 
                 if t["template_id"] == config.selected_template), 
                config.selected_template
            )
            summary_panel.append(f"   • Template: {template_name}\n\n")
        
        estimated_cost = self._estimate_generation_cost(config)
        summary_panel.append(f"💸 Estimated Generation Cost: ${estimated_cost:.2f}\n", style="cyan")
        
        panel = Panel(
            summary_panel,
            title="[bold]Final Configuration Summary[/bold]",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Final confirmation
        if not Confirm.ask("\n[bold]Is this configuration correct?[/bold]", default=True):
            # Allow going back to make changes
            self.console.print("[yellow]You can use the back navigation to make changes.[/yellow]")
            return False
        
        # Ready to generate
        self.console.print("\n[bold green]🎉 Configuration Complete![/bold green]")
        self.console.print("[dim]Ready to generate your security specification.[/dim]")
        
        return True
    
    def _advance_step(self, session: WizardSession) -> WizardSession:
        """Advance to next wizard step"""
        
        # Add current step to history
        session.step_history.append(session.current_step)
        
        # Determine next step
        step_order = [
            WizardStep.WELCOME,
            WizardStep.PROJECT_TYPE,
            WizardStep.PROJECT_DETAILS,
            WizardStep.SECURITY_REQUIREMENTS,
            WizardStep.AGENT_SELECTION,
            WizardStep.BUDGET_CONFIGURATION,
            WizardStep.TEMPLATE_RECOMMENDATION,
            WizardStep.FINAL_CONFIRMATION,
            WizardStep.COMPLETED
        ]
        
        current_index = step_order.index(session.current_step)
        if current_index < len(step_order) - 1:
            session.current_step = step_order[current_index + 1]
        
        return session
    
    def _go_back_step(self, session: WizardSession) -> WizardSession:
        """Go back to previous wizard step"""
        
        if session.step_history:
            session.current_step = session.step_history.pop()
        
        return session
    
    def _estimate_generation_cost(self, config: ProjectConfiguration) -> float:
        """Estimate specification generation cost"""
        
        # Base cost
        base_cost = 2.0
        
        # Agent cost multiplier
        agent_multiplier = len(config.selected_agents) * 0.5
        
        # Complexity multiplier based on requirements
        complexity_multiplier = 1.0
        if len(config.compliance_frameworks) > 2:
            complexity_multiplier += 0.3
        if len(config.security_priorities) > 4:
            complexity_multiplier += 0.2
        if config.expected_users > 10000:
            complexity_multiplier += 0.2
        
        # Quality multiplier
        quality_multipliers = {
            "speed": 0.8,
            "balanced": 1.0,
            "quality": 1.4
        }
        quality_multiplier = quality_multipliers.get(config.quality_preference, 1.0)
        
        return base_cost + agent_multiplier * complexity_multiplier * quality_multiplier
    
    def _should_auto_save(self, session: WizardSession) -> bool:
        """Check if session should be auto-saved"""
        
        if not session.last_saved_at:
            return True
        
        time_since_save = (datetime.utcnow() - session.last_saved_at).total_seconds()
        return time_since_save > self.save_interval
    
    def _save_session(self, session: WizardSession):
        """Save wizard session to file"""
        
        session_file = self.sessions_dir / f"{session.session_id}.json"
        session.save_to_file(session_file)
        
        # Keep only last 10 sessions
        self._cleanup_old_sessions()
    
    def _cleanup_old_sessions(self):
        """Cleanup old wizard sessions"""
        
        session_files = list(self.sessions_dir.glob("wizard_*.json"))
        if len(session_files) > 10:
            # Sort by modification time and remove oldest
            session_files.sort(key=lambda f: f.stat().st_mtime)
            for old_file in session_files[:-10]:
                old_file.unlink()
    
    def list_saved_sessions(self) -> List[Dict[str, str]]:
        """List saved wizard sessions"""
        
        sessions = []
        for session_file in self.sessions_dir.glob("wizard_*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                sessions.append({
                    'session_id': data['session_id'],
                    'project_name': data['configuration'].get('project_name', 'Unnamed'),
                    'current_step': data['current_step'].replace('_', ' ').title(),
                    'created': data['started_at'][:16],
                    'status': data['configuration'].get('completion_status', 'in_progress')
                })
            except:
                continue
        
        return sorted(sessions, key=lambda x: x['created'], reverse=True)


# Convenience function for CLI integration
def run_specification_wizard(resume_session_id: Optional[str] = None) -> Optional[ProjectConfiguration]:
    """Run the interactive specification wizard"""
    wizard = SpecificationWizard()
    return wizard.start_wizard(resume_session_id)