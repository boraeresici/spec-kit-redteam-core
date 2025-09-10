#!/usr/bin/env python3
"""
Spec-Kit RED TEAM Plugin

Multi-Agent RED TEAM Collaborative AI Plugin for Spec-Kit
Works with official upstream spec-kit from https://github.com/github/spec-kit
"""

__version__ = "1.0.0"
__plugin_name__ = "redteam-collaborative-ai"
__requires__ = ["spec-kit>=1.0.0"]

def register_plugin(main_app):
    """
    Smart plugin registration with upstream core detection
    
    This function handles:
    1. Core availability detection
    2. Compatibility checking
    3. Automatic installation guidance
    4. Fallback mechanisms
    """
    from .core_manager import core_manager
    from rich.console import Console
    
    console = Console()
    
    # Check if core is available and compatible
    if not core_manager.is_core_available():
        # Core is missing or incompatible
        core_info = core_manager.get_core_info()
        
        if not core_info["installed"]:
            console.print("[yellow]⚠️  Spec-Kit core not found![/yellow]")
            console.print("[cyan]The RED TEAM plugin requires spec-kit core.[/cyan]")
            console.print("\n[bold]Quick fix:[/bold]")
            console.print("[cyan]specify collab install-core[/cyan]")
            
        elif not core_info["compatible"]:
            console.print(f"[yellow]⚠️  Spec-Kit core v{core_info['version']} is not compatible![/yellow]")
            console.print(f"[cyan]Plugin requires spec-kit v{__requires__[0].split('>=')[1]}+[/cyan]")
            console.print("\n[bold]Quick fix:[/bold]")
            console.print("[cyan]specify collab update-core[/cyan]")
            
        elif not core_info["import_available"]:
            console.print("[yellow]⚠️  Spec-Kit core installation appears corrupted![/yellow]")
            console.print("\n[bold]Quick fix:[/bold]")
            console.print("[cyan]specify collab repair[/cyan]")
        
        # Register a minimal fallback command for diagnostics
        try:
            from .commands.diagnostics import app as diagnostics_app
            main_app.add_typer(
                diagnostics_app,
                name="collab",
                help="RED TEAM Collaborative AI (Setup Required)"
            )
            return True
        except ImportError:
            # Even fallback failed, register nothing
            return False
    
    # Core is available and compatible, register normally
    try:
        from .commands.collaborative import app as collaborative_app
        main_app.add_typer(
            collaborative_app,
            name="collab", 
            help="Multi-Agent RED TEAM Collaborative AI"
        )
        
        # Success message
        core_info = core_manager.get_core_info()
        console.print(f"[green]✓[/green] RED TEAM Collaborative AI plugin loaded (core v{core_info['version']})")
        return True
        
    except ImportError as e:
        console.print(f"[red]✗ Failed to load RED TEAM plugin: {e}[/red]")
        console.print("[cyan]Try: specify collab doctor[/cyan]")
        return False

def check_compatibility():
    """Check compatibility with current spec-kit core version"""
    from .core_manager import core_manager
    return core_manager.is_core_available()

def get_plugin_info():
    """Get plugin information for diagnostics"""
    from .core_manager import core_manager
    
    return {
        "plugin_version": __version__,
        "plugin_name": __plugin_name__,
        "requires": __requires__,
        "core_info": core_manager.get_core_info(),
        "compatible": core_manager.is_core_available()
    }