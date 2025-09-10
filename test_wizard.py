#!/usr/bin/env python3
"""
Test specification wizard functionality

Quick test of wizard components and user experience
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from spec_kit_redteam_plugin.commands.specification_wizard import (
        SpecificationWizard, WizardStep, ProjectConfiguration, WizardSession
    )
    from spec_kit_redteam_plugin.commands.template_cli import specification_wizard
    
    print("✅ Wizard imports successful")
    
    # Test wizard initialization
    wizard = SpecificationWizard()
    print("✅ Wizard initialization successful")
    
    # Test session creation
    session = wizard._create_new_session()
    print(f"✅ Session creation successful: {session.session_id}")
    
    # Test configuration
    config = ProjectConfiguration()
    config.project_name = "Test Project"
    config.project_type = "web"
    config.selected_agents = ["pm", "technical", "security"]
    config.max_budget = 25.0
    
    print("✅ Configuration creation successful")
    
    # Test cost estimation
    estimated_cost = wizard._estimate_generation_cost(config)
    print(f"✅ Cost estimation: ${estimated_cost:.2f}")
    
    # Test agent selection
    agents = config.selected_agents  
    print(f"✅ Agent selection: {agents}")
    
    # Test session saving (create directories if needed)
    wizard.sessions_dir.mkdir(parents=True, exist_ok=True)
    session.configuration = config
    wizard._save_session(session)
    print("✅ Session saving successful")
    
    # Test session listing
    sessions = wizard.list_saved_sessions()
    print(f"✅ Session listing: Found {len(sessions)} sessions")
    
    print(f"\n🎉 All wizard tests passed!")
    print(f"📁 Sessions saved in: {wizard.sessions_dir}")
    print(f"💡 To test interactively, run: python -m spec_kit_redteam_plugin.commands.template_cli wizard")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()