#!/usr/bin/env python3
"""
RedTeam Template Engine - Core Implementation

Provides template-based security specification generation with built-in
security frameworks and customizable project templates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from enum import Enum
import yaml
import json
from datetime import datetime
import hashlib
import re


class TemplateComplexity(Enum):
    """Template complexity levels affecting cost and generation time."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class TemplateCategory(Enum):
    """Template categories for different project types."""
    WEB_APPLICATION = "web"
    REST_API = "api"
    MOBILE_APP = "mobile"
    MICROSERVICE = "microservice"
    INFRASTRUCTURE = "infrastructure"
    DESKTOP_APP = "desktop"
    IOT_DEVICE = "iot"


class SecurityFramework(Enum):
    """Supported security frameworks and standards."""
    OWASP = "OWASP"
    NIST = "NIST"
    ISO27001 = "ISO27001"
    SOC2 = "SOC2"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    API_SECURITY = "API_Security"
    CONTAINER_SECURITY = "Container_Security"


class TierRequired(Enum):
    """Subscription tier required to access template."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class ValidationRule:
    """Template validation rule definition."""
    field: str
    rule_type: str  # "required", "min_length", "regex", "enum"
    value: Any
    message: str


@dataclass
class TemplateMetadata:
    """Template metadata and configuration."""
    created_at: datetime
    updated_at: datetime
    version: str
    author: str
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    rating: float = 0.0
    reviews_count: int = 0


@dataclass
class SecurityTemplate:
    """Core security template definition."""
    
    # Identity (required fields first)
    id: str
    name: str
    description: str
    
    # Categorization (required fields)
    category: TemplateCategory
    security_frameworks: List[SecurityFramework]
    complexity_level: TemplateComplexity
    
    # Agent Configuration (required fields)
    required_agents: List[str]
    
    # Cost & Performance (required fields)
    estimated_cost: float
    estimated_time_minutes: int
    
    # Template Content (required fields)
    template_content: str
    
    # Optional fields with defaults
    tier_required: TierRequired = TierRequired.FREE
    optional_agents: List[str] = field(default_factory=list)
    recommended_agents: List[str] = field(default_factory=list)
    prompt_template: str = ""
    context_variables: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    metadata: TemplateMetadata = field(default_factory=lambda: TemplateMetadata(
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        version="1.0.0",
        author="RedTeam System"
    ))
    
    def __post_init__(self):
        """Validate template after initialization."""
        if not self.id or not self.name:
            raise ValueError("Template ID and name are required")
        
        if not self.template_content:
            raise ValueError("Template content cannot be empty")
        
        # Ensure required agents are valid
        valid_agents = {"pm", "technical", "security", "qa", "compliance", "devops"}
        invalid_agents = set(self.required_agents) - valid_agents
        if invalid_agents:
            raise ValueError(f"Invalid agent types: {invalid_agents}")
    
    def get_all_agents(self) -> List[str]:
        """Get all agents (required + optional + recommended)."""
        all_agents = set(self.required_agents)
        all_agents.update(self.optional_agents)
        all_agents.update(self.recommended_agents)
        return list(all_agents)
    
    def calculate_cost_estimate(self, selected_agents: List[str]) -> float:
        """Calculate cost estimate based on selected agents."""
        base_cost = self.estimated_cost
        
        # Complexity multiplier
        complexity_multipliers = {
            TemplateComplexity.LOW: 0.7,
            TemplateComplexity.MEDIUM: 1.0,
            TemplateComplexity.HIGH: 1.5,
            TemplateComplexity.CRITICAL: 2.0
        }
        
        complexity_cost = base_cost * complexity_multipliers[self.complexity_level]
        
        # Agent count multiplier
        agent_count = len(selected_agents)
        if agent_count <= 2:
            agent_multiplier = 1.0
        elif agent_count <= 4:
            agent_multiplier = 1.3
        else:
            agent_multiplier = 1.6
        
        return complexity_cost * agent_multiplier
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate template configuration against rules."""
        errors = []
        
        for rule in self.validation_rules:
            field_value = config.get(rule.field)
            
            if rule.rule_type == "required" and not field_value:
                errors.append(f"{rule.field}: {rule.message}")
            elif rule.rule_type == "min_length" and len(str(field_value or "")) < rule.value:
                errors.append(f"{rule.field}: {rule.message}")
            elif rule.rule_type == "regex" and field_value:
                if not re.match(rule.value, str(field_value)):
                    errors.append(f"{rule.field}: {rule.message}")
            elif rule.rule_type == "enum" and field_value not in rule.value:
                errors.append(f"{rule.field}: {rule.message}")
        
        return errors


class TemplateValidationError(Exception):
    """Raised when template validation fails."""
    pass


class TemplateNotFoundError(Exception):
    """Raised when requested template is not found."""
    pass


class TemplateAccessDeniedError(Exception):
    """Raised when user doesn't have access to premium template."""
    pass


class TemplateManager:
    """Manages security templates and provides template operations."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "built_in"
        self.templates: Dict[str, SecurityTemplate] = {}
        self.user_templates: Dict[str, SecurityTemplate] = {}
        self.template_index: Dict[str, Set[str]] = {
            "category": {},
            "frameworks": {},
            "complexity": {},
            "agents": {}
        }
        
        # Load built-in templates
        self._load_built_in_templates()
    
    def _load_built_in_templates(self):
        """Load built-in templates from YAML files."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                
                template = self._create_template_from_dict(template_data)
                self.templates[template.id] = template
                self._update_index(template)
                
            except Exception as e:
                print(f"Failed to load template {template_file}: {e}")
    
    def _create_template_from_dict(self, data: Dict[str, Any]) -> SecurityTemplate:
        """Create SecurityTemplate from dictionary data."""
        
        # Parse enums
        category = TemplateCategory(data.get("category", "web"))
        complexity = TemplateComplexity(data.get("complexity_level", "medium"))
        tier_required = TierRequired(data.get("tier_required", "free"))
        
        frameworks = []
        for fw in data.get("security_frameworks", []):
            try:
                frameworks.append(SecurityFramework(fw))
            except ValueError:
                print(f"Unknown security framework: {fw}")
        
        # Parse validation rules
        validation_rules = []
        for rule_data in data.get("validation_rules", []):
            rule = ValidationRule(
                field=rule_data["field"],
                rule_type=rule_data["rule_type"],
                value=rule_data["value"],
                message=rule_data["message"]
            )
            validation_rules.append(rule)
        
        # Create metadata
        metadata = TemplateMetadata(
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "RedTeam System"),
            tags=data.get("tags", [])
        )
        
        return SecurityTemplate(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=category,
            security_frameworks=frameworks,
            complexity_level=complexity,
            tier_required=tier_required,
            required_agents=data.get("required_agents", []),
            optional_agents=data.get("optional_agents", []),
            recommended_agents=data.get("recommended_agents", []),
            estimated_cost=float(data.get("estimated_cost", 5.0)),
            estimated_time_minutes=int(data.get("estimated_time_minutes", 60)),
            template_content=data.get("template_content", ""),
            prompt_template=data.get("prompt_template", ""),
            context_variables=data.get("context_variables", {}),
            validation_rules=validation_rules,
            metadata=metadata
        )
    
    def _update_index(self, template: SecurityTemplate):
        """Update search indexes for template."""
        # Category index
        category_key = template.category.value
        if category_key not in self.template_index["category"]:
            self.template_index["category"][category_key] = set()
        self.template_index["category"][category_key].add(template.id)
        
        # Frameworks index
        for framework in template.security_frameworks:
            fw_key = framework.value
            if fw_key not in self.template_index["frameworks"]:
                self.template_index["frameworks"][fw_key] = set()
            self.template_index["frameworks"][fw_key].add(template.id)
        
        # Complexity index
        complexity_key = template.complexity_level.value
        if complexity_key not in self.template_index["complexity"]:
            self.template_index["complexity"][complexity_key] = set()
        self.template_index["complexity"][complexity_key].add(template.id)
        
        # Agents index
        for agent in template.get_all_agents():
            if agent not in self.template_index["agents"]:
                self.template_index["agents"][agent] = set()
            self.template_index["agents"][agent].add(template.id)
    
    def get_template(self, template_id: str, user_tier: str = "free") -> SecurityTemplate:
        """Get template by ID with tier access checking."""
        template = self.templates.get(template_id) or self.user_templates.get(template_id)
        
        if not template:
            raise TemplateNotFoundError(f"Template '{template_id}' not found")
        
        # Check tier access
        tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}
        required_level = tier_hierarchy[template.tier_required.value]
        user_level = tier_hierarchy.get(user_tier, 0)
        
        if user_level < required_level:
            raise TemplateAccessDeniedError(
                f"Template '{template_id}' requires {template.tier_required.value} subscription"
            )
        
        return template
    
    def list_templates(self, 
                      category: Optional[str] = None,
                      framework: Optional[str] = None,
                      complexity: Optional[str] = None,
                      user_tier: str = "free") -> List[SecurityTemplate]:
        """List templates with optional filtering."""
        
        # Start with all template IDs
        template_ids = set(self.templates.keys())
        
        # Apply filters
        if category and category in self.template_index["category"]:
            template_ids &= self.template_index["category"][category]
        
        if framework and framework in self.template_index["frameworks"]:
            template_ids &= self.template_index["frameworks"][framework]
        
        if complexity and complexity in self.template_index["complexity"]:
            template_ids &= self.template_index["complexity"][complexity]
        
        # Filter by user tier and build result list
        result = []
        tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}
        user_level = tier_hierarchy.get(user_tier, 0)
        
        for template_id in template_ids:
            template = self.templates[template_id]
            required_level = tier_hierarchy[template.tier_required.value]
            
            if user_level >= required_level:
                result.append(template)
        
        # Sort by usage count and rating
        result.sort(key=lambda t: (t.metadata.usage_count, t.metadata.rating), reverse=True)
        
        return result
    
    def search_templates(self, query: str, user_tier: str = "free") -> List[SecurityTemplate]:
        """Search templates by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            score = 0
            
            # Name match (highest score)
            if query_lower in template.name.lower():
                score += 10
            
            # Description match
            if query_lower in template.description.lower():
                score += 5
            
            # Tags match
            for tag in template.metadata.tags:
                if query_lower in tag.lower():
                    score += 3
            
            # Framework match
            for framework in template.security_frameworks:
                if query_lower in framework.value.lower():
                    score += 7
            
            if score > 0:
                # Check tier access
                tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}
                required_level = tier_hierarchy[template.tier_required.value]
                user_level = tier_hierarchy.get(user_tier, 0)
                
                if user_level >= required_level:
                    results.append((template, score))
        
        # Sort by score and return templates
        results.sort(key=lambda x: x[1], reverse=True)
        return [template for template, score in results]
    
    def add_template(self, template: SecurityTemplate, user_template: bool = True):
        """Add a new template."""
        if user_template:
            self.user_templates[template.id] = template
        else:
            self.templates[template.id] = template
        
        self._update_index(template)
    
    def validate_template(self, template_data: Dict[str, Any]) -> List[str]:
        """Validate template data structure."""
        errors = []
        
        required_fields = ["id", "name", "description", "category", "template_content"]
        for field in required_fields:
            if not template_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate enum values
        if "category" in template_data:
            try:
                TemplateCategory(template_data["category"])
            except ValueError:
                valid_categories = [c.value for c in TemplateCategory]
                errors.append(f"Invalid category. Valid options: {valid_categories}")
        
        if "complexity_level" in template_data:
            try:
                TemplateComplexity(template_data["complexity_level"])
            except ValueError:
                valid_complexities = [c.value for c in TemplateComplexity]
                errors.append(f"Invalid complexity. Valid options: {valid_complexities}")
        
        # Validate agents
        valid_agents = {"pm", "technical", "security", "qa", "compliance", "devops"}
        for agent_list_key in ["required_agents", "optional_agents", "recommended_agents"]:
            if agent_list_key in template_data:
                invalid_agents = set(template_data[agent_list_key]) - valid_agents
                if invalid_agents:
                    errors.append(f"Invalid agents in {agent_list_key}: {invalid_agents}")
        
        return errors
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about available templates."""
        total_templates = len(self.templates)
        
        # Count by category
        category_counts = {}
        for template in self.templates.values():
            cat = template.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Count by tier
        tier_counts = {}
        for template in self.templates.values():
            tier = template.tier_required.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Count by framework
        framework_counts = {}
        for template in self.templates.values():
            for framework in template.security_frameworks:
                fw = framework.value
                framework_counts[fw] = framework_counts.get(fw, 0) + 1
        
        return {
            "total_templates": total_templates,
            "by_category": category_counts,
            "by_tier": tier_counts,
            "by_framework": framework_counts,
            "average_cost": sum(t.estimated_cost for t in self.templates.values()) / total_templates if total_templates > 0 else 0
        }