#!/usr/bin/env python3
"""
Test script for RedTeam Template System

Tests the core functionality of the template engine, recommendation system,
and CLI integration.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.spec_kit_redteam_plugin.templates.template_engine import (
    TemplateManager, SecurityTemplate, TemplateCategory, 
    SecurityFramework, TemplateComplexity, TierRequired
)
from src.spec_kit_redteam_plugin.templates.recommendation_engine import (
    TemplateRecommendationEngine, ProjectAnalysis
)


def test_template_loading():
    """Test template loading and validation."""
    print("🔧 Testing template loading...")
    
    template_manager = TemplateManager()
    
    # Test template listing
    templates = template_manager.list_templates()
    print(f"✅ Loaded {len(templates)} templates")
    
    # Test specific template access
    for template_id in ["owasp-web-app", "basic-api-security", "microservice-security", "mobile-app-security"]:
        try:
            template = template_manager.get_template(template_id)
            print(f"✅ Template '{template_id}': {template.name}")
            print(f"   Category: {template.category.value}, Complexity: {template.complexity_level.value}")
            print(f"   Agents: {', '.join(template.required_agents)}")
            print(f"   Cost: ${template.estimated_cost:.2f}")
        except Exception as e:
            print(f"❌ Failed to load template '{template_id}': {e}")
    
    return len(templates) > 0


def test_template_search():
    """Test template search functionality."""
    print("\n🔍 Testing template search...")
    
    template_manager = TemplateManager()
    
    test_queries = [
        "web application",
        "api security", 
        "microservice",
        "mobile app",
        "owasp"
    ]
    
    for query in test_queries:
        results = template_manager.search_templates(query)
        print(f"✅ Query '{query}': {len(results)} results")
        if results:
            print(f"   Top result: {results[0].name}")
    
    return True


def test_recommendation_engine():
    """Test AI-powered template recommendation."""
    print("\n🤖 Testing recommendation engine...")
    
    template_manager = TemplateManager()
    recommendation_engine = TemplateRecommendationEngine(template_manager)
    
    test_descriptions = [
        "Build a secure web application with user authentication and payment processing",
        "Create a REST API for mobile app backend with OAuth2 authentication",
        "Develop a microservice architecture for e-commerce platform",
        "Build a mobile application for iOS and Android with biometric authentication",
        "Create a healthcare management system with HIPAA compliance"
    ]
    
    for description in test_descriptions:
        print(f"\n📝 Description: {description[:60]}...")
        
        # Test project analysis
        analysis = recommendation_engine.analyze_project_description(description)
        print(f"   Project Type: {analysis.project_type.value} (confidence: {analysis.confidence:.2f})")
        print(f"   Data Sensitivity: {analysis.data_sensitivity.value}")
        print(f"   Compliance: {', '.join(analysis.compliance_indicators) if analysis.compliance_indicators else 'None'}")
        
        # Test recommendations
        recommendations = recommendation_engine.recommend_templates(description, max_recommendations=2)
        print(f"   Recommendations: {len(recommendations)}")
        
        for i, rec in enumerate(recommendations, 1):
            confidence_pct = int(rec.confidence_score * 100)
            print(f"   {i}. {rec.template.name} ({confidence_pct}% match, ${rec.estimated_cost:.2f})")
    
    return True


def test_template_validation():
    """Test template validation system."""
    print("\n✅ Testing template validation...")
    
    template_manager = TemplateManager()
    
    # Test valid template data
    valid_template_data = {
        "id": "test-template",
        "name": "Test Template",
        "description": "A test template for validation",
        "category": "web",
        "template_content": "This is test content",
        "required_agents": ["pm", "technical"],
        "estimated_cost": 5.0
    }
    
    errors = template_manager.validate_template(valid_template_data)
    if not errors:
        print("✅ Valid template data passed validation")
    else:
        print(f"❌ Valid template failed validation: {errors}")
    
    # Test invalid template data
    invalid_template_data = {
        "id": "",  # Invalid: empty ID
        "name": "Test Template",
        "category": "invalid_category",  # Invalid category
        "required_agents": ["invalid_agent"],  # Invalid agent
        "estimated_cost": -5.0  # Would be caught by SecurityTemplate validation
    }
    
    errors = template_manager.validate_template(invalid_template_data)
    if errors:
        print(f"✅ Invalid template correctly failed validation: {len(errors)} errors")
    else:
        print("❌ Invalid template incorrectly passed validation")
    
    return True


def test_template_statistics():
    """Test template statistics functionality."""
    print("\n📊 Testing template statistics...")
    
    template_manager = TemplateManager()
    stats = template_manager.get_template_stats()
    
    print(f"✅ Total templates: {stats['total_templates']}")
    print(f"✅ Average cost: ${stats['average_cost']:.2f}")
    print(f"✅ Categories: {list(stats['by_category'].keys())}")
    print(f"✅ Tiers: {list(stats['by_tier'].keys())}")
    print(f"✅ Frameworks: {list(stats['by_framework'].keys())}")
    
    return stats['total_templates'] > 0


def test_tier_access_control():
    """Test subscription tier access control."""
    print("\n🔒 Testing tier access control...")
    
    template_manager = TemplateManager()
    
    # Test free tier access
    free_templates = template_manager.list_templates(user_tier="free")
    print(f"✅ Free tier access: {len(free_templates)} templates")
    
    # Test pro tier access
    pro_templates = template_manager.list_templates(user_tier="pro")
    print(f"✅ Pro tier access: {len(pro_templates)} templates")
    
    # Test enterprise tier access
    enterprise_templates = template_manager.list_templates(user_tier="enterprise")
    print(f"✅ Enterprise tier access: {len(enterprise_templates)} templates")
    
    # Verify hierarchy (enterprise >= pro >= free)
    assert len(enterprise_templates) >= len(pro_templates) >= len(free_templates)
    print("✅ Tier hierarchy is correct")
    
    return True


def run_all_tests():
    """Run all template system tests."""
    print("🚀 Starting RedTeam Template System Tests\n")
    
    tests = [
        ("Template Loading", test_template_loading),
        ("Template Search", test_template_search),
        ("Recommendation Engine", test_recommendation_engine),
        ("Template Validation", test_template_validation),
        ("Template Statistics", test_template_statistics),
        ("Tier Access Control", test_tier_access_control),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"=" * 60)
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            failed += 1
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Template system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)