#!/usr/bin/env python3
"""
Advanced Error Handler for Spec-Kit RED TEAM Plugin

Provides user-friendly error messages with context, suggestions, and recovery options.
"""

import sys
import traceback
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    API = "api"
    PLUGIN = "plugin"
    SPEC_KIT = "spec_kit"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class PluginError(Exception):
    """Base exception for plugin errors with enhanced context"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PLUGIN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        suggestions: Optional[List[str]] = None,
        technical_details: Optional[str] = None,
        recovery_actions: Optional[List[str]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.technical_details = technical_details
        self.recovery_actions = recovery_actions or []
        self.error_code = error_code or f"{category.value.upper()}_{severity.value.upper()}"


class ErrorHandler:
    """Advanced error handling with user-friendly messages and recovery suggestions"""
    
    def __init__(self, console: Optional[Console] = None, debug_mode: bool = False):
        self.console = console or Console()
        self.debug_mode = debug_mode
        self.error_history: List[Dict[str, Any]] = []
        
        # Error message templates
        self.error_templates = {
            ErrorCategory.SPEC_KIT: {
                "not_found": {
                    "title": "🔧 Spec-Kit Not Found",
                    "message": "Spec-Kit core is not installed or cannot be found",
                    "suggestions": [
                        "Run: specify collab install-core",
                        "Manual install: pip install git+https://github.com/github/spec-kit.git",
                        "Check your Python environment"
                    ],
                    "recovery_actions": ["install-core", "doctor"]
                },
                "version_incompatible": {
                    "title": "⚠️ Version Compatibility Issue", 
                    "message": "Installed Spec-Kit version is not compatible with this plugin",
                    "suggestions": [
                        "Update Spec-Kit: specify collab update-core",
                        "Check version requirements",
                        "Contact support if issue persists"
                    ],
                    "recovery_actions": ["update-core", "repair"]
                }
            },
            ErrorCategory.NETWORK: {
                "connection_failed": {
                    "title": "🌐 Network Connection Failed",
                    "message": "Unable to connect to AI service",
                    "suggestions": [
                        "Check your internet connection",
                        "Verify API endpoints are accessible",
                        "Try again in a few moments",
                        "Check firewall/proxy settings"
                    ],
                    "recovery_actions": ["retry", "check-network"]
                },
                "timeout": {
                    "title": "⏰ Request Timeout",
                    "message": "AI service request timed out",
                    "suggestions": [
                        "Try with a smaller request",
                        "Check your network speed",
                        "Increase timeout settings",
                        "Retry the operation"
                    ],
                    "recovery_actions": ["retry", "adjust-timeout"]
                }
            },
            ErrorCategory.API: {
                "rate_limit": {
                    "title": "🚦 API Rate Limit Exceeded",
                    "message": "Too many requests to AI service",
                    "suggestions": [
                        "Wait a few minutes before retrying",
                        "Consider upgrading your API plan", 
                        "Use caching to reduce requests",
                        "Spread out your requests over time"
                    ],
                    "recovery_actions": ["wait", "enable-cache"]
                },
                "auth_failed": {
                    "title": "🔐 Authentication Failed",
                    "message": "Invalid or missing API credentials",
                    "suggestions": [
                        "Check your API key configuration",
                        "Verify API key is still valid",
                        "Set environment variables correctly",
                        "Contact API provider if key issues persist"
                    ],
                    "recovery_actions": ["check-auth", "reconfigure"]
                }
            },
            ErrorCategory.USER_INPUT: {
                "invalid_command": {
                    "title": "❌ Invalid Command",
                    "message": "Command or parameters are not recognized",
                    "suggestions": [
                        "Check command syntax: specify collab --help",
                        "Verify all required parameters",
                        "Use quotes around descriptions with spaces",
                        "See usage examples: specify collab agents"
                    ],
                    "recovery_actions": ["help", "examples"]
                },
                "missing_params": {
                    "title": "📋 Missing Parameters", 
                    "message": "Required parameters are missing",
                    "suggestions": [
                        "Check required parameters with --help",
                        "Ensure all mandatory fields are provided",
                        "Use proper command format"
                    ],
                    "recovery_actions": ["help", "validate-params"]
                }
            }
        }
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None,
        show_technical: bool = None
    ) -> bool:
        """
        Handle error with user-friendly display and recovery options
        
        Returns:
            bool: True if error was handled gracefully, False if critical
        """
        # Determine if we should show technical details
        if show_technical is None:
            show_technical = self.debug_mode or isinstance(error, PluginError)
        
        # Log error to history
        self._log_error(error, context)
        
        if isinstance(error, PluginError):
            return self._handle_plugin_error(error, context, show_technical)
        else:
            return self._handle_generic_error(error, context, show_technical)
    
    def _handle_plugin_error(
        self, 
        error: PluginError, 
        context: Optional[Dict[str, Any]], 
        show_technical: bool
    ) -> bool:
        """Handle PluginError with rich formatting"""
        
        # Get error template
        template = self._get_error_template(error.category, error.error_code)
        
        # Create main panel content
        content = Text()
        
        # Error message
        content.append(f"{error.message}\n\n", style="bold red")
        
        # Technical details if available and requested
        if show_technical and error.technical_details:
            content.append("Technical Details:\n", style="bold yellow")
            content.append(f"{error.technical_details}\n\n", style="dim")
        
        # Context information
        if context:
            content.append("Context:\n", style="bold blue")
            for key, value in context.items():
                content.append(f"  {key}: {value}\n", style="dim")
            content.append("\n")
        
        # Create suggestions table
        suggestions_table = None
        if error.suggestions or template.get("suggestions"):
            suggestions = error.suggestions or template.get("suggestions", [])
            suggestions_table = Table(title="💡 Suggestions", show_header=False, box=None)
            suggestions_table.add_column("", style="cyan")
            
            for i, suggestion in enumerate(suggestions, 1):
                suggestions_table.add_row(f"{i}. {suggestion}")
        
        # Create recovery actions table  
        recovery_table = None
        if error.recovery_actions or template.get("recovery_actions"):
            actions = error.recovery_actions or template.get("recovery_actions", [])
            recovery_table = Table(title="🔧 Quick Actions", show_header=False, box=None)
            recovery_table.add_column("", style="green")
            
            action_commands = {
                "install-core": "specify collab install-core",
                "update-core": "specify collab update-core", 
                "repair": "specify collab repair",
                "doctor": "specify collab doctor",
                "help": "specify collab --help",
                "retry": "↑ (Press up arrow to retry last command)",
                "check-auth": "Check your API configuration",
                "examples": "specify collab agents  # See available agents"
            }
            
            for action in actions:
                command = action_commands.get(action, action)
                recovery_table.add_row(f"• {command}")
        
        # Display main error panel
        title = template.get("title", f"❌ {error.category.value.title()} Error")
        panel = Panel(
            content,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Display suggestions and recovery in columns
        if suggestions_table or recovery_table:
            columns_content = []
            if suggestions_table:
                columns_content.append(suggestions_table)
            if recovery_table:
                columns_content.append(recovery_table)
            
            if columns_content:
                self.console.print(Columns(columns_content, equal=True, expand=True))
        
        # Return based on severity
        return error.severity != ErrorSeverity.CRITICAL
    
    def _handle_generic_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]], 
        show_technical: bool
    ) -> bool:
        """Handle generic Python exceptions"""
        
        error_name = error.__class__.__name__
        
        # Create content
        content = Text()
        content.append(f"An unexpected error occurred: {str(error)}\n\n", style="bold red")
        
        if show_technical:
            content.append("Technical Details:\n", style="bold yellow")
            content.append(f"Error Type: {error_name}\n", style="dim")
            if context:
                content.append("Context: ", style="dim")
                content.append(f"{json.dumps(context, indent=2)}\n", style="dim")
            content.append(f"\nTraceback:\n{traceback.format_exc()}", style="dim")
        else:
            content.append("For technical details, run with --verbose flag\n", style="dim")
        
        # Generic recovery actions
        recovery_table = Table(title="🔧 Recovery Actions", show_header=False, box=None)
        recovery_table.add_column("", style="green")
        recovery_table.add_row("• specify collab doctor  # Diagnose issues")
        recovery_table.add_row("• specify collab --help  # See available commands")
        recovery_table.add_row("• Check GitHub issues for similar problems")
        
        # Display
        panel = Panel(
            content,
            title="[bold red]🐛 Unexpected Error[/bold red]",
            border_style="red", 
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print(recovery_table)
        
        return False  # Generic errors are considered critical
    
    def _get_error_template(self, category: ErrorCategory, error_code: str) -> Dict[str, Any]:
        """Get error template for category and code"""
        category_templates = self.error_templates.get(category, {})
        
        # Try to find specific error template
        for template_key, template in category_templates.items():
            if template_key in error_code.lower():
                return template
        
        # Return empty template if not found
        return {}
    
    def _log_error(self, error: Exception, context: Optional[Dict[str, Any]]):
        """Log error to history for debugging"""
        error_entry = {
            "timestamp": str(Path().absolute()),  # Simple timestamp
            "type": error.__class__.__name__,
            "message": str(error),
            "context": context
        }
        
        self.error_history.append(error_entry)
        
        # Keep only last 50 errors
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]
    
    def show_error_history(self, limit: int = 10):
        """Show recent error history"""
        if not self.error_history:
            self.console.print("📊 No errors recorded yet", style="green")
            return
        
        table = Table(title=f"🐛 Recent Errors (Last {limit})")
        table.add_column("Type", style="red")
        table.add_column("Message", style="white")
        table.add_column("Context", style="dim")
        
        recent_errors = self.error_history[-limit:]
        for error in recent_errors:
            context_str = str(error.get("context", ""))[:50] + "..." if error.get("context") else ""
            table.add_row(
                error["type"],
                error["message"][:80] + "..." if len(error["message"]) > 80 else error["message"],
                context_str
            )
        
        self.console.print(table)


# Global error handler instance
error_handler = ErrorHandler()


# Common error factory functions
def spec_kit_not_found_error(details: str = None) -> PluginError:
    """Create spec-kit not found error"""
    return PluginError(
        message="Spec-Kit core is required but not found",
        category=ErrorCategory.SPEC_KIT,
        severity=ErrorSeverity.ERROR,
        technical_details=details,
        error_code="SPEC_KIT_NOT_FOUND"
    )


def network_error(message: str, details: str = None) -> PluginError:
    """Create network-related error"""
    return PluginError(
        message=f"Network error: {message}",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.WARNING,
        technical_details=details,
        error_code="NETWORK_CONNECTION_FAILED"
    )


def api_error(message: str, status_code: int = None, details: str = None) -> PluginError:
    """Create API-related error"""
    error_code = "API_RATE_LIMIT" if status_code == 429 else "API_AUTH_FAILED" if status_code == 401 else "API_ERROR"
    
    return PluginError(
        message=f"API error: {message}",
        category=ErrorCategory.API,
        severity=ErrorSeverity.ERROR if status_code in [401, 403] else ErrorSeverity.WARNING,
        technical_details=f"Status Code: {status_code}\n{details}" if status_code else details,
        error_code=error_code
    )


def user_input_error(message: str, suggestions: List[str] = None) -> PluginError:
    """Create user input validation error"""
    return PluginError(
        message=message,
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        suggestions=suggestions,
        error_code="USER_INPUT_INVALID_COMMAND"
    )