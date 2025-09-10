#!/usr/bin/env python3
"""
Template Recommendation Engine

AI-powered template recommendation system that analyzes project descriptions
and requirements to suggest the most appropriate security templates.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
import math
from collections import Counter

from .template_engine import SecurityTemplate, TemplateManager, TemplateCategory, SecurityFramework


@dataclass
class TemplateRecommendation:
    """Template recommendation with confidence score and reasoning."""
    template: SecurityTemplate
    confidence_score: float  # 0.0 to 1.0
    reasoning: List[str]
    estimated_cost: float
    match_factors: Dict[str, float]


class ProjectType(Enum):
    """Project type classification for recommendation."""
    WEB_APPLICATION = "web_app"
    REST_API = "api"
    MOBILE_APP = "mobile"
    MICROSERVICE = "microservice"
    DESKTOP_APP = "desktop"
    IOT_DEVICE = "iot"
    INFRASTRUCTURE = "infrastructure"
    DATABASE = "database"
    BLOCKCHAIN = "blockchain"
    UNKNOWN = "unknown"


class DataSensitivity(Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class ProjectAnalysis:
    """Analysis results of a project description."""
    project_type: ProjectType
    confidence: float
    keywords: List[str]
    technologies: List[str]
    security_keywords: List[str]
    data_sensitivity: DataSensitivity
    compliance_indicators: List[str]
    complexity_indicators: List[str]


class TemplateRecommendationEngine:
    """AI-powered template recommendation engine."""
    
    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        
        # Technology keywords for project type classification
        self.tech_keywords = {
            ProjectType.WEB_APPLICATION: [
                "web", "website", "webapp", "html", "css", "javascript", "react", 
                "angular", "vue", "django", "flask", "express", "nodejs", "php",
                "laravel", "symfony", "asp.net", "mvc", "frontend", "backend"
            ],
            ProjectType.REST_API: [
                "api", "rest", "restful", "endpoint", "json", "xml", "http",
                "graphql", "grpc", "swagger", "openapi", "microapi", "service"
            ],
            ProjectType.MOBILE_APP: [
                "mobile", "ios", "android", "app", "smartphone", "tablet",
                "react-native", "flutter", "xamarin", "ionic", "swift", "kotlin",
                "objective-c", "java", "cordova", "phonegap"
            ],
            ProjectType.MICROSERVICE: [
                "microservice", "microservices", "docker", "kubernetes", "k8s",
                "container", "containerized", "service-mesh", "istio", "linkerd",
                "distributed", "cloud-native", "serverless", "lambda"
            ],
            ProjectType.DESKTOP_APP: [
                "desktop", "windows", "macos", "linux", "gui", "electron",
                "wpf", "winforms", "qt", "gtk", "swing", "javafx", "tkinter"
            ],
            ProjectType.IOT_DEVICE: [
                "iot", "internet-of-things", "embedded", "sensor", "device",
                "arduino", "raspberry-pi", "esp32", "mqtt", "coap", "zigbee"
            ],
            ProjectType.INFRASTRUCTURE: [
                "infrastructure", "cloud", "aws", "azure", "gcp", "terraform",
                "ansible", "puppet", "chef", "cloudformation", "helm", "iac"
            ]
        }
        
        # Security-related keywords
        self.security_keywords = [
            "authentication", "authorization", "encryption", "security", "secure",
            "privacy", "gdpr", "hipaa", "pci", "compliance", "audit", "logging",
            "monitoring", "firewall", "vpn", "ssl", "tls", "certificate",
            "oauth", "saml", "jwt", "session", "csrf", "xss", "sql-injection",
            "vulnerability", "penetration", "threat", "risk", "incident"
        ]
        
        # Compliance framework indicators
        self.compliance_indicators = {
            "GDPR": ["gdpr", "eu", "europe", "privacy", "personal data", "data protection"],
            "HIPAA": ["hipaa", "healthcare", "medical", "patient", "phi", "health"],
            "PCI-DSS": ["pci", "payment", "credit card", "financial", "merchant"],
            "SOC2": ["soc2", "service organization", "trust services", "audit"],
            "ISO27001": ["iso27001", "iso 27001", "information security", "isms"],
            "NIST": ["nist", "cybersecurity framework", "federal", "government"]
        }
        
        # Data sensitivity indicators
        self.sensitivity_indicators = {
            DataSensitivity.RESTRICTED: [
                "classified", "top secret", "restricted", "highly sensitive",
                "national security", "defense", "military"
            ],
            DataSensitivity.CONFIDENTIAL: [
                "confidential", "sensitive", "private", "proprietary", "trade secret",
                "financial", "medical", "personal", "pii", "phi"
            ],
            DataSensitivity.INTERNAL: [
                "internal", "company", "corporate", "business", "employee"
            ],
            DataSensitivity.PUBLIC: [
                "public", "open", "community", "marketing", "website"
            ]
        }
    
    def analyze_project_description(self, description: str, 
                                  additional_context: Optional[Dict[str, Any]] = None) -> ProjectAnalysis:
        """Analyze project description to extract key characteristics."""
        description_lower = description.lower()
        
        # Extract keywords
        words = re.findall(r'\b\w+\b', description_lower)
        word_freq = Counter(words)
        
        # Classify project type
        project_type, type_confidence = self._classify_project_type(description_lower, word_freq)
        
        # Extract technologies
        technologies = self._extract_technologies(description_lower)
        
        # Identify security keywords
        security_keywords = [word for word in words if word in self.security_keywords]
        
        # Determine data sensitivity
        data_sensitivity = self._determine_data_sensitivity(description_lower)
        
        # Find compliance indicators
        compliance_indicators = self._find_compliance_indicators(description_lower)
        
        # Assess complexity
        complexity_indicators = self._assess_complexity(description_lower, word_freq)
        
        return ProjectAnalysis(
            project_type=project_type,
            confidence=type_confidence,
            keywords=list(word_freq.keys())[:20],  # Top 20 keywords
            technologies=technologies,
            security_keywords=security_keywords,
            data_sensitivity=data_sensitivity,
            compliance_indicators=compliance_indicators,
            complexity_indicators=complexity_indicators
        )
    
    def _classify_project_type(self, description: str, word_freq: Counter) -> Tuple[ProjectType, float]:
        """Classify project type based on description."""
        type_scores = {}
        
        for project_type, keywords in self.tech_keywords.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword in description:
                    # Weight by frequency and keyword importance
                    freq_weight = word_freq.get(keyword, 0)
                    importance_weight = 1.0
                    
                    # Some keywords are more definitive
                    if keyword in ["api", "mobile", "microservice", "iot"]:
                        importance_weight = 2.0
                    
                    score += freq_weight * importance_weight
                    matches += 1
            
            # Normalize score by number of possible keywords
            if matches > 0:
                type_scores[project_type] = score / len(keywords) * math.log(matches + 1)
        
        if not type_scores:
            return ProjectType.UNKNOWN, 0.0
        
        # Find highest scoring type
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # Calculate confidence (normalize to 0-1 range)
        total_score = sum(type_scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0
        
        return best_type, min(confidence, 1.0)
    
    def _extract_technologies(self, description: str) -> List[str]:
        """Extract mentioned technologies from description."""
        technologies = []
        
        # Common technology patterns
        tech_patterns = [
            r'\b(react|angular|vue|django|flask|express|spring|laravel)\b',
            r'\b(docker|kubernetes|aws|azure|gcp|terraform)\b',
            r'\b(mysql|postgresql|mongodb|redis|elasticsearch)\b',
            r'\b(python|java|javascript|typescript|go|rust|php)\b',
            r'\b(ios|android|swift|kotlin|flutter|react-native)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            technologies.extend(matches)
        
        return list(set(technologies))
    
    def _determine_data_sensitivity(self, description: str) -> DataSensitivity:
        """Determine data sensitivity level from description."""
        sensitivity_scores = {}
        
        for sensitivity, indicators in self.sensitivity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in description)
            if score > 0:
                sensitivity_scores[sensitivity] = score
        
        if not sensitivity_scores:
            return DataSensitivity.INTERNAL  # Default
        
        return max(sensitivity_scores, key=sensitivity_scores.get)
    
    def _find_compliance_indicators(self, description: str) -> List[str]:
        """Find compliance framework indicators in description."""
        found_frameworks = []
        
        for framework, indicators in self.compliance_indicators.items():
            if any(indicator in description for indicator in indicators):
                found_frameworks.append(framework)
        
        return found_frameworks
    
    def _assess_complexity(self, description: str, word_freq: Counter) -> List[str]:
        """Assess project complexity indicators."""
        complexity_indicators = []
        
        # High complexity indicators
        high_complexity_words = [
            "distributed", "scalable", "high-availability", "fault-tolerant",
            "real-time", "streaming", "big-data", "machine-learning", "ai",
            "blockchain", "cryptocurrency", "multi-tenant", "enterprise"
        ]
        
        # Medium complexity indicators
        medium_complexity_words = [
            "authentication", "authorization", "payment", "integration",
            "workflow", "reporting", "analytics", "search"
        ]
        
        high_matches = sum(1 for word in high_complexity_words if word in description)
        medium_matches = sum(1 for word in medium_complexity_words if word in description)
        
        if high_matches >= 2:
            complexity_indicators.append("high_complexity")
        elif high_matches >= 1 or medium_matches >= 3:
            complexity_indicators.append("medium_complexity")
        else:
            complexity_indicators.append("low_complexity")
        
        # Specific complexity factors
        if len(description) > 500:
            complexity_indicators.append("detailed_requirements")
        
        if any(word in description for word in ["integrate", "third-party", "external"]):
            complexity_indicators.append("integration_complexity")
        
        return complexity_indicators
    
    def recommend_templates(self, description: str, 
                          user_tier: str = "free",
                          max_recommendations: int = 3,
                          additional_context: Optional[Dict[str, Any]] = None) -> List[TemplateRecommendation]:
        """Generate template recommendations based on project description."""
        
        # Analyze the project description
        analysis = self.analyze_project_description(description, additional_context)
        
        # Get all available templates for user tier
        available_templates = self.template_manager.list_templates(user_tier=user_tier)
        
        # Score each template
        template_scores = []
        
        for template in available_templates:
            score_data = self._score_template(template, analysis)
            if score_data["total_score"] > 0.1:  # Minimum threshold
                
                recommendation = TemplateRecommendation(
                    template=template,
                    confidence_score=score_data["total_score"],
                    reasoning=score_data["reasoning"],
                    estimated_cost=template.calculate_cost_estimate(
                        template.recommended_agents or template.required_agents
                    ),
                    match_factors=score_data["factors"]
                )
                template_scores.append(recommendation)
        
        # Sort by confidence score and return top recommendations
        template_scores.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return template_scores[:max_recommendations]
    
    def _score_template(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """Score a template against project analysis."""
        factors = {}
        reasoning = []
        
        # 1. Project type match (40% weight)
        type_score = self._score_project_type_match(template, analysis)
        factors["project_type"] = type_score
        if type_score > 0.7:
            reasoning.append(f"Strong project type match ({template.category.value})")
        elif type_score > 0.3:
            reasoning.append(f"Partial project type match ({template.category.value})")
        
        # 2. Security framework alignment (25% weight)
        framework_score = self._score_framework_alignment(template, analysis)
        factors["frameworks"] = framework_score
        if framework_score > 0.5:
            frameworks = [f.value for f in template.security_frameworks]
            reasoning.append(f"Matches security frameworks: {', '.join(frameworks)}")
        
        # 3. Complexity match (20% weight)
        complexity_score = self._score_complexity_match(template, analysis)
        factors["complexity"] = complexity_score
        if complexity_score > 0.6:
            reasoning.append(f"Appropriate complexity level ({template.complexity_level.value})")
        
        # 4. Technology alignment (10% weight)
        tech_score = self._score_technology_alignment(template, analysis)
        factors["technology"] = tech_score
        if tech_score > 0.3:
            reasoning.append("Technology stack alignment")
        
        # 5. Security requirements match (5% weight)
        security_score = self._score_security_requirements(template, analysis)
        factors["security"] = security_score
        if security_score > 0.4:
            reasoning.append("Strong security requirements match")
        
        # Calculate weighted total score
        weights = {
            "project_type": 0.40,
            "frameworks": 0.25,
            "complexity": 0.20,
            "technology": 0.10,
            "security": 0.05
        }
        
        total_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        # Boost score for exact matches
        if type_score > 0.9:
            total_score *= 1.2
        
        # Apply data sensitivity bonus
        if self._has_appropriate_data_protection(template, analysis):
            total_score *= 1.1
            reasoning.append("Appropriate data protection measures")
        
        return {
            "total_score": min(total_score, 1.0),
            "factors": factors,
            "reasoning": reasoning
        }
    
    def _score_project_type_match(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> float:
        """Score project type alignment."""
        # Direct category match
        category_mapping = {
            ProjectType.WEB_APPLICATION: TemplateCategory.WEB_APPLICATION,
            ProjectType.REST_API: TemplateCategory.REST_API,
            ProjectType.MOBILE_APP: TemplateCategory.MOBILE_APP,
            ProjectType.MICROSERVICE: TemplateCategory.MICROSERVICE,
            ProjectType.DESKTOP_APP: TemplateCategory.DESKTOP_APP,
            ProjectType.IOT_DEVICE: TemplateCategory.IOT_DEVICE,
            ProjectType.INFRASTRUCTURE: TemplateCategory.INFRASTRUCTURE
        }
        
        if analysis.project_type in category_mapping:
            if template.category == category_mapping[analysis.project_type]:
                return 1.0 * analysis.confidence
        
        # Partial matches
        partial_matches = {
            (ProjectType.REST_API, TemplateCategory.WEB_APPLICATION): 0.7,
            (ProjectType.WEB_APPLICATION, TemplateCategory.REST_API): 0.6,
            (ProjectType.MICROSERVICE, TemplateCategory.REST_API): 0.8,
            (ProjectType.MOBILE_APP, TemplateCategory.REST_API): 0.5,
        }
        
        match_key = (analysis.project_type, template.category)
        if match_key in partial_matches:
            return partial_matches[match_key] * analysis.confidence
        
        return 0.0
    
    def _score_framework_alignment(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> float:
        """Score security framework alignment."""
        if not analysis.compliance_indicators:
            return 0.3  # Neutral score if no specific compliance mentioned
        
        template_frameworks = [f.value for f in template.security_frameworks]
        matches = sum(1 for indicator in analysis.compliance_indicators 
                     if any(indicator.upper() in fw.upper() for fw in template_frameworks))
        
        if matches == 0:
            return 0.2
        
        # Score based on match ratio
        match_ratio = matches / len(analysis.compliance_indicators)
        return min(match_ratio * 1.5, 1.0)
    
    def _score_complexity_match(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> float:
        """Score complexity level alignment."""
        complexity_mapping = {
            "low_complexity": ["low"],
            "medium_complexity": ["medium"],
            "high_complexity": ["high", "critical"]
        }
        
        template_complexity = template.complexity_level.value
        
        for indicator in analysis.complexity_indicators:
            if indicator in complexity_mapping:
                if template_complexity in complexity_mapping[indicator]:
                    return 1.0
                elif abs(["low", "medium", "high"].index(template_complexity) - 
                        ["low", "medium", "high"].index(complexity_mapping[indicator][0])) <= 1:
                    return 0.6
        
        return 0.3  # Default moderate match
    
    def _score_technology_alignment(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> float:
        """Score technology stack alignment."""
        # This is basic - could be enhanced with more sophisticated tech matching
        tech_score = 0.0
        
        if analysis.technologies:
            # Boost score if template mentions compatible technologies
            tech_score = 0.5
        
        return tech_score
    
    def _score_security_requirements(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> float:
        """Score security requirements alignment."""
        if not analysis.security_keywords:
            return 0.2
        
        security_keyword_count = len(analysis.security_keywords)
        
        # More security keywords indicate higher security focus
        if security_keyword_count >= 5:
            return 1.0
        elif security_keyword_count >= 3:
            return 0.7
        elif security_keyword_count >= 1:
            return 0.4
        
        return 0.2
    
    def _has_appropriate_data_protection(self, template: SecurityTemplate, analysis: ProjectAnalysis) -> bool:
        """Check if template has appropriate data protection for sensitivity level."""
        # Higher sensitivity data should use more comprehensive templates
        if analysis.data_sensitivity in [DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED]:
            return template.complexity_level.value in ["high", "critical"]
        
        return True
    
    def get_recommendation_explanation(self, recommendation: TemplateRecommendation) -> str:
        """Generate human-readable explanation for recommendation."""
        template = recommendation.template
        confidence_pct = int(recommendation.confidence_score * 100)
        
        explanation = f"""
Template: {template.name}
Confidence: {confidence_pct}%
Estimated Cost: ${recommendation.estimated_cost:.2f}

Why this template was recommended:
"""
        
        for reason in recommendation.reasoning:
            explanation += f"• {reason}\n"
        
        explanation += f"""
Template includes:
• {len(template.required_agents)} required agents: {', '.join(template.required_agents)}
• Security frameworks: {', '.join([f.value for f in template.security_frameworks])}
• Complexity level: {template.complexity_level.value}
• Estimated time: {template.estimated_time_minutes} minutes
"""
        
        return explanation