#!/usr/bin/env python3
"""
Version Management and Backward Compatibility for Spec-Kit RED TEAM Plugin

Handles plugin versioning, compatibility checks, and smooth upgrades.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from packaging import version

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class CompatibilityLevel(Enum):
    """Compatibility levels for version checking"""
    FULL = "full"           # Fully compatible
    PARTIAL = "partial"     # Some features may not work
    DEPRECATED = "deprecated" # Old version, still works but deprecated
    INCOMPATIBLE = "incompatible"  # Not compatible


@dataclass
class VersionInfo:
    """Version information structure"""
    version: str
    release_date: Optional[str] = None
    breaking_changes: List[str] = None
    new_features: List[str] = None
    deprecated_features: List[str] = None
    min_spec_kit_version: Optional[str] = None
    max_spec_kit_version: Optional[str] = None


class VersionManager:
    """Manages plugin versioning and compatibility"""
    
    # Plugin version history and compatibility matrix
    VERSION_HISTORY = {
        "1.0.0": VersionInfo(
            version="1.0.0",
            release_date="2024-01-15",
            new_features=[
                "Initial RED TEAM collaborative AI plugin",
                "Multi-agent security analysis",
                "Spec-Kit integration",
                "Budget management",
                "Error handling system"
            ],
            min_spec_kit_version="1.0.0",
            max_spec_kit_version="1.9.999"
        ),
        "1.1.0": VersionInfo(
            version="1.1.0", 
            release_date="2024-02-01",
            new_features=[
                "Enhanced error messages",
                "Plugin versioning system",
                "Better input validation",
                "Improved diagnostics"
            ],
            min_spec_kit_version="1.0.0",
            max_spec_kit_version="2.0.0"
        ),
        "2.0.0": VersionInfo(
            version="2.0.0",
            release_date="2024-03-01",
            breaking_changes=[
                "Command structure changed from 'redteam' to 'collab'",
                "New error handling format",
                "Updated configuration format"
            ],
            new_features=[
                "Async/await agent communication",
                "Memory management improvements", 
                "Hot reload capabilities",
                "Error recovery system"
            ],
            deprecated_features=[
                "Old command format (still supported with warnings)"
            ],
            min_spec_kit_version="1.5.0",
            max_spec_kit_version="2.5.0"
        )
    }
    
    # Current plugin version
    CURRENT_VERSION = "1.1.0"
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.config_path = Path.home() / ".config" / "spec-kit-redteam" / "version.json"
        self.user_config = self._load_user_config()
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user's version configuration"""
        if not self.config_path.exists():
            return {"installed_version": None, "last_check": None, "skip_warnings": []}
        
        try:
            return json.loads(self.config_path.read_text())
        except Exception:
            return {"installed_version": None, "last_check": None, "skip_warnings": []}
    
    def _save_user_config(self):
        """Save user configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.user_config, indent=2))
    
    def get_current_version(self) -> str:
        """Get current plugin version"""
        return self.CURRENT_VERSION
    
    def get_version_info(self, version_str: str) -> Optional[VersionInfo]:
        """Get version information for a specific version"""
        return self.VERSION_HISTORY.get(version_str)
    
    def check_spec_kit_compatibility(self, spec_kit_version: str) -> Tuple[CompatibilityLevel, str]:
        """
        Check compatibility with spec-kit version
        
        Returns:
            Tuple of (compatibility_level, explanation)
        """
        current_info = self.get_version_info(self.CURRENT_VERSION)
        if not current_info:
            return CompatibilityLevel.INCOMPATIBLE, "Unable to determine version compatibility"
        
        try:
            spec_kit_v = version.parse(spec_kit_version)
            min_v = version.parse(current_info.min_spec_kit_version) if current_info.min_spec_kit_version else None
            max_v = version.parse(current_info.max_spec_kit_version) if current_info.max_spec_kit_version else None
            
            # Check if within supported range
            if min_v and spec_kit_v < min_v:
                return CompatibilityLevel.INCOMPATIBLE, f"Spec-Kit v{spec_kit_version} is too old. Minimum required: v{current_info.min_spec_kit_version}"
            
            if max_v and spec_kit_v > max_v:
                return CompatibilityLevel.PARTIAL, f"Spec-Kit v{spec_kit_version} is newer than tested. Some features may not work correctly."
            
            return CompatibilityLevel.FULL, f"Spec-Kit v{spec_kit_version} is fully compatible"
            
        except Exception as e:
            return CompatibilityLevel.INCOMPATIBLE, f"Unable to parse version: {e}"
    
    def check_plugin_upgrade(self) -> Optional[Tuple[str, VersionInfo]]:
        """
        Check if plugin upgrade is available
        
        Returns:
            Tuple of (available_version, version_info) or None if up to date
        """
        # In a real implementation, this would check PyPI or GitHub releases
        # For now, simulate with version history
        
        current_v = version.parse(self.CURRENT_VERSION)
        latest_version = None
        latest_info = None
        
        for ver_str, info in self.VERSION_HISTORY.items():
            try:
                ver = version.parse(ver_str)
                if ver > current_v:
                    if latest_version is None or ver > version.parse(latest_version):
                        latest_version = ver_str
                        latest_info = info
            except Exception:
                continue
        
        if latest_version:
            return latest_version, latest_info
        
        return None
    
    def show_version_info(self, include_history: bool = False):
        """Display current version information"""
        current_info = self.get_version_info(self.CURRENT_VERSION)
        
        # Main version panel
        version_text = Text()
        version_text.append("🔴 Multi-Agent RED TEAM Plugin\n", style="bold red")
        version_text.append(f"Version: ", style="white")
        version_text.append(f"{self.CURRENT_VERSION}", style="bold green")
        
        if current_info and current_info.release_date:
            version_text.append(f"\nReleased: {current_info.release_date}", style="dim")
        
        # Compatibility info
        version_text.append(f"\n\nSpec-Kit Compatibility:", style="bold yellow")
        if current_info:
            if current_info.min_spec_kit_version:
                version_text.append(f"\n  Minimum: v{current_info.min_spec_kit_version}", style="cyan")
            if current_info.max_spec_kit_version:
                version_text.append(f"\n  Maximum: v{current_info.max_spec_kit_version}", style="cyan")
        
        panel = Panel(
            version_text,
            title="[bold]📦 Plugin Information[/bold]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Features in current version
        if current_info:
            if current_info.new_features:
                features_table = Table(title="✨ Features in Current Version")
                features_table.add_column("Feature", style="green")
                
                for feature in current_info.new_features:
                    features_table.add_row(f"• {feature}")
                
                self.console.print(features_table)
        
        # Check for updates
        upgrade_info = self.check_plugin_upgrade()
        if upgrade_info:
            available_version, version_info = upgrade_info
            
            update_text = Text()
            update_text.append(f"🎉 Update Available: v{available_version}\n", style="bold green")
            
            if version_info.new_features:
                update_text.append("New Features:\n", style="bold yellow")
                for feature in version_info.new_features:
                    update_text.append(f"  • {feature}\n", style="white")
            
            if version_info.breaking_changes:
                update_text.append("\n⚠️  Breaking Changes:\n", style="bold red")
                for change in version_info.breaking_changes:
                    update_text.append(f"  • {change}\n", style="yellow")
            
            update_text.append(f"\nTo upgrade: pip install --upgrade spec-kit-redteam-plugin", style="cyan")
            
            update_panel = Panel(
                update_text,
                title="[bold green]📦 Update Available[/bold green]",
                border_style="green",
                padding=(1, 2)
            )
            
            self.console.print(update_panel)
        else:
            self.console.print("\n✅ [green]Plugin is up to date[/green]")
        
        # Version history
        if include_history:
            self.show_version_history()
    
    def show_version_history(self):
        """Show version history"""
        history_table = Table(title="📚 Version History")
        history_table.add_column("Version", style="cyan")
        history_table.add_column("Release Date", style="dim")
        history_table.add_column("Key Changes", style="white")
        
        # Sort versions by release date (newest first)
        sorted_versions = sorted(
            self.VERSION_HISTORY.items(),
            key=lambda x: x[1].release_date or "1900-01-01",
            reverse=True
        )
        
        for ver_str, info in sorted_versions:
            key_changes = []
            
            if info.breaking_changes:
                key_changes.extend([f"💥 {change}" for change in info.breaking_changes[:2]])
            
            if info.new_features:
                key_changes.extend([f"✨ {feature}" for feature in info.new_features[:2]])
            
            if info.deprecated_features:
                key_changes.extend([f"⚠️ {feature}" for feature in info.deprecated_features[:1]])
            
            changes_text = "\n".join(key_changes[:3])
            if len(key_changes) > 3:
                changes_text += f"\n... and {len(key_changes) - 3} more"
            
            # Mark current version
            version_display = ver_str
            if ver_str == self.CURRENT_VERSION:
                version_display = f"{ver_str} (current)"
            
            history_table.add_row(
                version_display,
                info.release_date or "Unknown",
                changes_text
            )
        
        self.console.print(history_table)
    
    def check_backward_compatibility(self, old_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if old configuration is compatible and provide migration warnings
        
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []
        
        # Check for deprecated command formats
        if "redteam_commands" in old_config:
            warnings.append(
                "⚠️ 'redteam' commands are deprecated. Use 'collab' commands instead."
            )
        
        # Check for old error handling format
        if "old_error_format" in old_config:
            warnings.append(
                "⚠️ Old error handling format detected. Error messages will use new format."
            )
        
        # Check for configuration structure changes
        if "version" not in old_config:
            warnings.append(
                "⚠️ Configuration missing version info. Some features may not work as expected."
            )
        
        # Most configurations should be backward compatible with warnings
        is_compatible = True
        
        return is_compatible, warnings
    
    def migrate_configuration(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate old configuration to new format
        
        Returns:
            Updated configuration dictionary
        """
        new_config = old_config.copy()
        
        # Add version info if missing
        if "version" not in new_config:
            new_config["version"] = self.CURRENT_VERSION
        
        # Migrate command format
        if "redteam_commands" in new_config:
            # Convert old redteam commands to collab commands
            old_commands = new_config.pop("redteam_commands", {})
            new_config["collab_commands"] = old_commands
        
        # Migrate error handling settings
        if "old_error_format" in new_config:
            new_config.pop("old_error_format")
            new_config["enhanced_errors"] = True
        
        # Update deprecated settings
        deprecated_keys = ["legacy_mode", "old_ui_format"]
        for key in deprecated_keys:
            if key in new_config:
                new_config.pop(key)
        
        return new_config
    
    def show_migration_guide(self, old_version: str, new_version: str):
        """Show migration guide between versions"""
        old_info = self.get_version_info(old_version)
        new_info = self.get_version_info(new_version)
        
        migration_text = Text()
        migration_text.append(f"🔄 Migration Guide: v{old_version} → v{new_version}\n\n", style="bold cyan")
        
        if new_info and new_info.breaking_changes:
            migration_text.append("💥 Breaking Changes:\n", style="bold red")
            for change in new_info.breaking_changes:
                migration_text.append(f"  • {change}\n", style="yellow")
            migration_text.append("\n")
        
        if new_info and new_info.deprecated_features:
            migration_text.append("⚠️ Deprecated Features:\n", style="bold yellow")
            for feature in new_info.deprecated_features:
                migration_text.append(f"  • {feature}\n", style="dim")
            migration_text.append("\n")
        
        # Migration steps
        migration_text.append("📋 Migration Steps:\n", style="bold green")
        migration_text.append("  1. Backup your current configuration\n", style="white")
        migration_text.append("  2. Update plugin: pip install --upgrade spec-kit-redteam-plugin\n", style="white")
        migration_text.append("  3. Run: specify collab doctor  # Check for issues\n", style="white")
        migration_text.append("  4. Update any scripts using old commands\n", style="white")
        migration_text.append("  5. Test your workflows with new version\n", style="white")
        
        panel = Panel(
            migration_text,
            title="[bold]🔄 Migration Guide[/bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        
        self.console.print(panel)


# Global version manager instance
version_manager = VersionManager()


def check_version_compatibility() -> bool:
    """
    Quick version compatibility check for plugin initialization
    
    Returns:
        True if everything is compatible, False if issues found
    """
    try:
        # This would be called during plugin initialization
        # to ensure compatibility before loading
        
        # Check if user configuration needs migration
        user_config = version_manager.user_config
        
        if not user_config.get("version"):
            # First time setup - no issues
            user_config["version"] = version_manager.CURRENT_VERSION
            version_manager._save_user_config()
            return True
        
        installed_version = user_config.get("version")
        current_version = version_manager.CURRENT_VERSION
        
        if installed_version != current_version:
            # Version change detected - check compatibility
            is_compatible, warnings = version_manager.check_backward_compatibility(user_config)
            
            if warnings and "skip_version_warnings" not in user_config.get("skip_warnings", []):
                # Show warnings to user
                console = Console()
                console.print("\n⚠️ [yellow]Plugin version change detected[/yellow]")
                for warning in warnings:
                    console.print(f"  {warning}")
                console.print("\nRun 'specify collab version --help' for migration guide\n")
            
            # Migrate configuration
            if is_compatible:
                new_config = version_manager.migrate_configuration(user_config)
                version_manager.user_config = new_config
                version_manager._save_user_config()
        
        return True
        
    except Exception:
        # If version checking fails, don't block plugin loading
        return True