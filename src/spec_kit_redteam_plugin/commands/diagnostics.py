#!/usr/bin/env python3
"""
Diagnostic and repair commands for spec-kit collaborative plugin.

These commands are available even when core is not properly installed,
to help users troubleshoot and fix installation issues.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core_manager import core_manager

app = typer.Typer(name="collab", help="Collaborative AI (Setup Required)")
console = Console()


@app.command("doctor")
def doctor():
    """Diagnose spec-kit installation issues"""
    console.print(Panel.fit(
        "[bold cyan]🔧 Spec-Kit Collaborative AI - System Diagnosis[/bold cyan]",
        border_style="cyan"
    ))
    
    # Show detailed status
    core_manager.show_installation_status()
    
    # Get detailed info for recommendations
    info = core_manager.get_core_info()
    
    console.print("\n[bold]📋 Recommendations:[/bold]")
    
    if not info["installed"]:
        console.print("1. [yellow]Install spec-kit core:[/yellow]")
        console.print("   [cyan]specify collab install-core[/cyan]")
        console.print("   [dim]or manually: pip install git+https://github.com/github/spec-kit.git[/dim]")
    
    elif not info["compatible"]:
        console.print("1. [yellow]Update spec-kit core:[/yellow]")
        console.print("   [cyan]specify collab update-core[/cyan]")
        console.print(f"   [dim]Current: v{info['version']}, Need: v{core_manager.CORE_COMPATIBILITY['minimum_version']}+[/dim]")
    
    elif not info["import_available"]:
        console.print("1. [yellow]Repair installation:[/yellow]")
        console.print("   [cyan]specify collab repair[/cyan]")
        console.print("   [dim]This will reinstall spec-kit core[/dim]")
    
    else:
        console.print("✅ [green]All systems operational![/green]")
        console.print("   Try: [cyan]specify collab generate \"hello world\"[/cyan]")
    
    console.print("\n[bold]📞 Additional Help:[/bold]")
    console.print("• [cyan]specify collab status[/cyan] - Show current status")
    console.print("• [cyan]specify collab install-core --help[/cyan] - Installation options")
    console.print("• GitHub: [link]https://github.com/boraeresici/spec-kit-collab[/link]")


@app.command("status")
def status():
    """Show current installation status"""
    console.print(Panel.fit(
        "[bold cyan]📊 Installation Status[/bold cyan]",
        border_style="cyan"
    ))
    
    core_manager.show_installation_status()
    
    # Show plugin info
    from .. import get_plugin_info
    plugin_info = get_plugin_info()
    
    console.print(f"\n[bold]🔌 Plugin Information:[/bold]")
    console.print(f"Version: [cyan]{plugin_info['plugin_version']}[/cyan]")
    console.print(f"Compatible: [{'green' if plugin_info['compatible'] else 'red'}]{plugin_info['compatible']}[/{'green' if plugin_info['compatible'] else 'red'}]")


@app.command("install-core")
def install_core(
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall even if already installed"),
    interactive: bool = typer.Option(True, "--interactive/--non-interactive", help="Interactive installation")
):
    """Install spec-kit core from official GitHub repository"""
    
    console.print(Panel.fit(
        "[bold cyan]📦 Installing Spec-Kit Core[/bold cyan]\n"
        "[dim]From: https://github.com/github/spec-kit[/dim]",
        border_style="cyan"
    ))
    
    if not force and core_manager.is_core_available():
        info = core_manager.get_core_info()
        console.print(f"[green]✓ Compatible spec-kit core v{info['version']} already installed![/green]")
        console.print("Use --force to reinstall anyway.")
        return
    
    if interactive and not force:
        if not typer.confirm("Install/update spec-kit core from GitHub?"):
            console.print("[yellow]Installation cancelled.[/yellow]")
            return
    
    # Perform installation
    with console.status("[cyan]Installing spec-kit core..."):
        success, message = core_manager.install_core_from_github(force=force)
    
    if success:
        console.print(f"[green]✅ {message}[/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("• [cyan]specify plugins[/cyan] - Verify plugin is loaded")
        console.print("• [cyan]specify collab generate \"hello world\"[/cyan] - Test functionality")
    else:
        console.print(f"[red]❌ {message}[/red]")
        console.print("\n[bold]Troubleshooting:[/bold]")
        console.print("• Check internet connection")
        console.print("• Verify pip is up to date: [cyan]pip install --upgrade pip[/cyan]")
        console.print("• Try manual installation: [cyan]pip install git+https://github.com/github/spec-kit.git[/cyan]")


@app.command("update-core")  
def update_core():
    """Update spec-kit core to latest compatible version"""
    
    console.print(Panel.fit(
        "[bold cyan]🔄 Updating Spec-Kit Core[/bold cyan]",
        border_style="cyan"
    ))
    
    info = core_manager.get_core_info()
    
    if not info["installed"]:
        console.print("[yellow]Spec-kit core is not installed.[/yellow]")
        console.print("Use [cyan]specify collab install-core[/cyan] instead.")
        return
    
    console.print(f"Current version: [cyan]v{info['version']}[/cyan]")
    console.print(f"Updating to latest from GitHub...")
    
    if typer.confirm("Continue with update?"):
        with console.status("[cyan]Updating spec-kit core..."):
            success, message = core_manager.install_core_from_github(force=True)
        
        if success:
            console.print(f"[green]✅ {message}[/green]")
        else:
            console.print(f"[red]❌ {message}[/red]")
    else:
        console.print("[yellow]Update cancelled.[/yellow]")


@app.command("repair")
def repair():
    """Attempt to repair corrupted spec-kit installation"""
    
    console.print(Panel.fit(
        "[bold cyan]🔧 Repairing Spec-Kit Installation[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print("[yellow]This will attempt to fix common installation issues...[/yellow]")
    
    if typer.confirm("Continue with repair?"):
        with console.status("[cyan]Repairing installation..."):
            success, message = core_manager.repair_installation()
        
        if success:
            console.print(f"[green]✅ {message}[/green]")
            console.print("\n[bold]Verification:[/bold]")
            console.print("Run [cyan]specify collab doctor[/cyan] to confirm repair.")
        else:
            console.print(f"[red]❌ {message}[/red]")
            console.print("\n[bold]Next steps:[/bold]")
            console.print("• Try: [cyan]specify collab install-core --force[/cyan]")
            console.print("• Manual: [cyan]pip uninstall spec-kit && pip install git+https://github.com/github/spec-kit.git[/cyan]")
    else:
        console.print("[yellow]Repair cancelled.[/yellow]")


@app.command("setup-guide")
def setup_guide():
    """Show detailed setup guide"""
    core_manager.show_installation_guide()


# Fallback commands that show helpful messages
@app.command("generate")  
def generate_fallback():
    """Generate collaborative specifications (requires setup)"""
    console.print("[yellow]⚠️  Collaborative AI requires setup![/yellow]")
    console.print("\n[bold]Quick setup:[/bold]")
    console.print("[cyan]specify collab install-core[/cyan]")
    console.print("\n[bold]Need help?[/bold]")
    console.print("[cyan]specify collab doctor[/cyan]")


@app.command("agents")
def agents_fallback():
    """List available collaborative agents (requires setup)"""
    console.print("[yellow]⚠️  Agent management requires setup![/yellow]")
    console.print("\n[bold]Quick setup:[/bold]")
    console.print("[cyan]specify collab install-core[/cyan]")


@app.command("estimate")
def estimate_fallback():
    """Estimate generation costs (requires setup)"""
    console.print("[yellow]⚠️  Cost estimation requires setup![/yellow]")
    console.print("\n[bold]Quick setup:[/bold]")
    console.print("[cyan]specify collab install-core[/cyan]")


if __name__ == "__main__":
    app()