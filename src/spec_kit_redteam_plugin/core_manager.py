#!/usr/bin/env python3
"""
Core Spec-Kit Manager

Manages detection, installation, and compatibility of upstream spec-kit core.
This allows our plugin to work seamlessly with the official GitHub repository.
"""

import subprocess
import sys
import importlib.util
from packaging import version
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Try modern importlib.metadata first, fallback to pkg_resources
try:
    from importlib.metadata import distribution, PackageNotFoundError
    def get_distribution(name):
        try:
            return distribution(name)
        except PackageNotFoundError:
            return None
except ImportError:
    # Python < 3.8 fallback
    try:
        import pkg_resources
        def get_distribution(name):
            try:
                return pkg_resources.get_distribution(name)
            except pkg_resources.DistributionNotFound:
                return None
    except ImportError:
        def get_distribution(name):
            return None

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class CoreSpecKitManager:
    """Manages spec-kit core installation and compatibility"""
    
    # Compatibility matrix
    CORE_COMPATIBILITY = {
        "minimum_version": "1.0.0",
        "maximum_version": "2.0.0",
        "recommended_version": "latest",
        "github_repo": "https://github.com/github/spec-kit.git"
    }
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._core_info_cache = None
    
    def get_core_info(self) -> Dict[str, Any]:
        """Get information about installed spec-kit core"""
        if self._core_info_cache:
            return self._core_info_cache
        
        info = {
            "installed": False,
            "version": None,
            "location": None,
            "compatible": False,
            "import_available": False
        }
        
        try:
            # Method 1: Try importing the module
            import specify_cli
            info["installed"] = True
            info["import_available"] = True
            info["version"] = getattr(specify_cli, '__version__', 'unknown')
            info["location"] = getattr(specify_cli, '__file__', 'unknown')
            
        except ImportError:
            # Method 2: Check via distribution metadata
            dist = get_distribution('spec-kit')
            if dist:
                info["installed"] = True
                info["version"] = dist.version
                info["location"] = getattr(dist, 'location', str(dist.files[0].parent) if hasattr(dist, 'files') and dist.files else 'unknown')
        
        # Check compatibility if version is available
        if info["version"] and info["version"] != 'unknown':
            info["compatible"] = self._is_version_compatible(info["version"])
        
        self._core_info_cache = info
        return info
    
    def _is_version_compatible(self, core_version: str) -> bool:
        """Check if core version is compatible with plugin"""
        try:
            v = version.parse(core_version)
            min_v = version.parse(self.CORE_COMPATIBILITY["minimum_version"])
            max_v = version.parse(self.CORE_COMPATIBILITY["maximum_version"])
            
            return min_v <= v < max_v
        except Exception:
            # If version parsing fails, assume incompatible
            return False
    
    def is_core_available(self) -> bool:
        """Check if compatible core is available"""
        info = self.get_core_info()
        return info["installed"] and info["compatible"] and info["import_available"]
    
    def install_core_from_github(self, force: bool = False) -> Tuple[bool, str]:
        """
        Install spec-kit core from official GitHub repository
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.is_core_available() and not force:
            return True, "Compatible core already installed"
        
        github_url = self.CORE_COMPATIBILITY["github_repo"]
        
        try:
            self.console.print(f"[cyan]Installing spec-kit core from:[/cyan] {github_url}")
            
            # Install from GitHub
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--upgrade",
                f"git+{github_url}"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clear cache to force re-detection
            self._core_info_cache = None
            
            # Verify installation
            if self.is_core_available():
                info = self.get_core_info()
                return True, f"Successfully installed spec-kit core v{info['version']}"
            else:
                return False, "Installation completed but core is not compatible"
                
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            
            # Enhanced error context
            if "Permission denied" in error_msg:
                return False, "Installation failed: Permission denied. Try running with sudo or check your Python environment permissions."
            elif "Connection" in error_msg or "network" in error_msg.lower():
                return False, "Installation failed: Network connection error. Check your internet connection and try again."
            elif "git" in error_msg.lower():
                return False, "Installation failed: Git is required but not found. Please install Git and try again."
            else:
                return False, f"Installation failed: {error_msg}"
                
        except Exception as e:
            return False, f"Unexpected error during installation: {str(e)}. Please try manual installation."
    
    def repair_installation(self) -> Tuple[bool, str]:
        """Attempt to repair core installation"""
        self.console.print("[yellow]Attempting to repair spec-kit installation...[/yellow]")
        
        # Strategy 1: Reinstall from GitHub
        success, message = self.install_core_from_github(force=True)
        if success:
            return True, f"Repair successful: {message}"
        
        # Strategy 2: Clear cache and retry detection
        self._core_info_cache = None
        if self.is_core_available():
            return True, "Repair successful: Core is now available"
        
        return False, f"Repair failed: {message}"
    
    def show_installation_status(self):
        """Display detailed installation status"""
        info = self.get_core_info()
        
        status_table = Table(title="🔧 Spec-Kit Core Status")
        status_table.add_column("Component", style="cyan", width=20)
        status_table.add_column("Status", width=15)
        status_table.add_column("Details", style="dim", width=40)
        
        # Installation status
        if info["installed"]:
            install_status = "[green]✓ Installed[/green]"
            install_details = f"Version: {info['version']}"
        else:
            install_status = "[red]✗ Not Found[/red]"
            install_details = "Run installation command"
        
        status_table.add_row("Installation", install_status, install_details)
        
        # Import status
        if info["import_available"]:
            import_status = "[green]✓ Import OK[/green]"
            import_details = "Module loads successfully"
        else:
            import_status = "[red]✗ Import Failed[/red]"
            import_details = "Module cannot be imported"
        
        status_table.add_row("Import", import_status, import_details)
        
        # Compatibility status
        if info["compatible"]:
            compat_status = "[green]✓ Compatible[/green]"
            compat_details = f"Supports plugin v1.0.0"
        else:
            compat_status = "[yellow]! Version Issue[/yellow]"
            compat_details = f"Need v{self.CORE_COMPATIBILITY['minimum_version']}+"
        
        status_table.add_row("Compatibility", compat_status, compat_details)
        
        self.console.print(status_table)
        
        # Show installation location if available
        if info["location"]:
            self.console.print(f"\n[dim]Installed at: {info['location']}[/dim]")
    
    def show_installation_guide(self):
        """Show user-friendly installation guide"""
        guide_content = Text()
        guide_content.append("🚀 ", style="bold")
        guide_content.append("Spec-Kit Core Installation Required\n\n", style="bold cyan")
        
        guide_content.append("The RED TEAM plugin requires spec-kit core package.\n\n", style="white")
        
        guide_content.append("📦 Automatic Installation:\n", style="bold green")
        guide_content.append("  specify collab install-core\n\n", style="cyan")
        
        guide_content.append("🔧 Manual Installation:\n", style="bold yellow")
        guide_content.append("  pip install git+https://github.com/github/spec-kit.git\n\n", style="cyan")
        
        guide_content.append("❓ Need Help?\n", style="bold magenta")
        guide_content.append("  specify collab doctor  # Diagnose installation issues\n", style="cyan")
        guide_content.append("  specify collab repair  # Auto-repair installation\n", style="cyan")
        
        panel = Panel(
            guide_content,
            title="[bold]🔌 Plugin Setup Guide[/bold]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def ensure_core_available(self, interactive: bool = True) -> bool:
        """
        Ensure compatible core is available, with user interaction if needed
        
        Args:
            interactive: If True, prompt user for permission to install
            
        Returns:
            True if core is available after this call
        """
        if self.is_core_available():
            return True
        
        info = self.get_core_info()
        
        if not info["installed"]:
            # Core not installed at all
            if interactive:
                self.console.print("[yellow]Spec-kit core not found![/yellow]")
                if self._prompt_install():
                    success, message = self.install_core_from_github()
                    if success:
                        self.console.print(f"[green]✓ {message}[/green]")
                        return True
                    else:
                        self.console.print(f"[red]✗ {message}[/red]")
                        self.show_installation_guide()
                        return False
                else:
                    self.show_installation_guide()
                    return False
            else:
                # Non-interactive mode, just return False
                return False
        
        elif not info["compatible"]:
            # Core installed but incompatible version
            if interactive:
                self.console.print(f"[yellow]Spec-kit core v{info['version']} is not compatible![/yellow]")
                self.console.print(f"[cyan]Need version {self.CORE_COMPATIBILITY['minimum_version']} or higher[/cyan]")
                
                if self._prompt_update():
                    success, message = self.install_core_from_github(force=True)
                    if success:
                        self.console.print(f"[green]✓ {message}[/green]")
                        return True
                    else:
                        self.console.print(f"[red]✗ {message}[/red]")
                        return False
                else:
                    self.show_installation_guide()
                    return False
            else:
                return False
        
        elif not info["import_available"]:
            # Core installed but cannot be imported
            if interactive:
                self.console.print("[yellow]Spec-kit core installation appears corrupted![/yellow]")
                if self._prompt_repair():
                    success, message = self.repair_installation()
                    if success:
                        self.console.print(f"[green]✓ {message}[/green]")
                        return True
                    else:
                        self.console.print(f"[red]✗ {message}[/red]")
                        return False
                else:
                    self.show_installation_guide()
                    return False
            else:
                return False
        
        return False
    
    def _prompt_install(self) -> bool:
        """Prompt user for permission to install core"""
        try:
            import typer
            return typer.confirm(
                "Would you like to install spec-kit core automatically?",
                default=True
            )
        except ImportError:
            # Fallback to input() if typer not available
            response = input("Install spec-kit core automatically? [Y/n]: ").strip().lower()
            return response in ('', 'y', 'yes')
    
    def _prompt_update(self) -> bool:
        """Prompt user for permission to update core"""
        try:
            import typer
            return typer.confirm(
                "Would you like to update spec-kit core to a compatible version?",
                default=True
            )
        except ImportError:
            response = input("Update spec-kit core to compatible version? [Y/n]: ").strip().lower()
            return response in ('', 'y', 'yes')
    
    def _prompt_repair(self) -> bool:
        """Prompt user for permission to repair installation"""
        try:
            import typer
            return typer.confirm(
                "Would you like to attempt automatic repair?",
                default=True
            )
        except ImportError:
            response = input("Attempt automatic repair? [Y/n]: ").strip().lower()
            return response in ('', 'y', 'yes')


# Global instance for easy access
core_manager = CoreSpecKitManager()