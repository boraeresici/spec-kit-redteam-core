# 🔴 Spec-Kit RED TEAM Core Plugin

> **Core CLI functionality for Multi-Agent RED TEAM Collaborative AI**

This is the core module of the RedTeam Multi-Agent system, providing CLI-based functionality for security analysis through collaborative AI agents.

## 🚀 Quick Start

### Standalone Installation
```bash
pip install spec-kit-redteam-plugin
specify collab generate "secure web application"
```

### Development Installation  
```bash
git clone https://github.com/boraeresici/spec-kit-redteam-core.git
cd spec-kit-redteam-core
pip install -e .
```

## 📦 Features

### ✅ Core Functionality (Free)
- **Basic Agent Communication** (PM + Technical agents)
- **Template System** with basic templates
- **Error Handling** with recovery suggestions
- **Plugin Versioning** and compatibility checks
- **Local File Operations** and caching

### 🔒 Limitations (Freemium Model)
```python
FREEMIUM_LIMITS = {
    "max_agents_per_generation": 2,
    "monthly_generations": 50,
    "template_access": "basic_only",
    "export_formats": ["markdown"],
    "advanced_features": False
}
```

## 🎯 Usage Examples

```bash
# Basic generation
specify collab generate "todo app with auth" --agent pm --agent technical

# Check usage
specify collab usage  # 47/50 monthly generations used

# Version info
specify collab version --history

# Error diagnostics
specify collab doctor
```

## 🔧 Available Commands

| Command | Description | Free/Pro |
|---------|-------------|----------|
| `generate` | Generate specifications | Free (limited) |
| `agents` | List available agents | Free |
| `templates` | Manage templates | Free (basic) |
| `version` | Version information | Free |
| `doctor` | System diagnostics | Free |
| `recovery-stats` | Recovery statistics | Free |

## 📈 Upgrade Options

When you hit freemium limits:

```bash
# Upgrade prompts
specify collab generate --agent security
# Output: "🔒 Security agent available in Pro. Upgrade: specify collab upgrade"

# Feature discovery
specify collab features
# Shows: Web Dashboard, Enterprise features, Advanced agents
```

## 🏗️ Architecture

This core module is designed to work standalone or as part of the larger RedTeam ecosystem:

```
RedTeam Ecosystem:
├── Core Plugin (this repo) - CLI functionality
├── Web Dashboard - Local UI enhancement
└── Enterprise SaaS - Cloud platform with team features
```

## 🔗 Integration

### With Web Dashboard
```bash
# Install web enhancement
pip install redteam-web-dashboard
specify collab ui --start  # Opens http://localhost:8080
```

### With Enterprise Platform
```bash
# Enterprise features
specify collab enterprise --signup
# Guides to cloud platform signup
```

## 📊 Performance

- **Generation Time**: 30-120 seconds (depending on complexity)
- **Cost per Generation**: $0.50-$3.00 (AI model costs)
- **Cache Hit Rate**: 30-50% for similar requests
- **Parallel Agents**: Up to 2 (free), 4+ (premium)

## 🛠️ Development

### Extension Points
```python
# Plugin extensions
from spec_kit_redteam_plugin import plugin_registry

@plugin_registry.register_agent
class CustomSecurityAgent(BaseAgent):
    def analyze(self, prompt):
        # Custom security analysis
        pass
```

### API Integration
```python
# Programmatic usage
from spec_kit_redteam_plugin import RedTeamCore

core = RedTeamCore()
result = await core.generate_spec(
    description="secure API",
    agents=["pm", "technical"],
    budget=10.0
)
```

## 📋 Requirements

- Python 3.11+
- Internet connection (for AI models)
- 50MB disk space
- API keys for AI services (handled automatically)

## 🆘 Support

- **Documentation**: [GitHub Wiki](https://github.com/boraeresici/spec-kit-redteam-core/wiki)
- **Issues**: [GitHub Issues](https://github.com/boraeresici/spec-kit-redteam-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/boraeresici/spec-kit-redteam-core/discussions)

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/boraeresici/spec-kit-redteam-core.git
cd spec-kit-redteam-core
pip install -e ".[dev]"
pytest tests/
```

---

**🔴 Part of the RedTeam Multi-Agent Ecosystem**

- **Core Plugin** (this repo): CLI functionality
- **[Web Dashboard](https://github.com/boraeresici/redteam-web-dashboard)**: Local web UI
- **Enterprise SaaS**: Cloud platform (private repo)