#!/usr/bin/env python3
"""
Technical Agent for collaborative specification generation.
Focuses on architecture, implementation feasibility, and technical constraints.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentResponse, AgentCapability


class TechnicalAgent(BaseAgent):
    """AI agent that thinks like a Technical Architect/Lead Developer"""
    
    def _load_configuration(self):
        """Load Technical-specific configuration"""
        self.focus_areas = [
            'architecture_design',
            'implementation_feasibility',
            'technical_constraints',
            'system_integration',
            'performance_requirements',
            'scalability_considerations'
        ]
        
        self.capabilities = [
            AgentCapability(
                name="analyze_technical_feasibility",
                description="Assess technical feasibility and complexity",
                input_format="Feature requirements and constraints",
                output_format="Technical analysis with implementation approach",
                estimated_tokens=2000
            ),
            AgentCapability(
                name="design_system_architecture",
                description="Design system architecture and component interactions",
                input_format="Requirements and existing system context",
                output_format="Architecture decisions and component design",
                estimated_tokens=1800
            ),
            AgentCapability(
                name="identify_technical_risks",
                description="Identify technical risks and mitigation strategies",
                input_format="Implementation plan and requirements",
                output_format="Risk assessment and mitigation recommendations",
                estimated_tokens=1200
            ),
            AgentCapability(
                name="validate_constitution_compliance",
                description="Ensure compliance with project constitution",
                input_format="Implementation plan and constitution",
                output_format="Compliance report and recommendations",
                estimated_tokens=1000
            )
        ]
        
        # Load constitution for compliance checking
        self.constitution = self._load_constitution()
    
    def _load_constitution(self) -> Dict:
        """Load project constitution for compliance checking"""
        # Look for constitution in expected locations
        constitution_paths = [
            Path.cwd() / "memory" / "constitution.md",
            Path.cwd() / "constitution.md",
            Path(__file__).parent.parent.parent / "memory" / "constitution.md"
        ]
        
        for path in constitution_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        # For now, store as raw text
                        # In production, this would be parsed into structured rules
                        return {'content': f.read(), 'path': str(path)}
                except Exception:
                    continue
        
        # Default constitution principles if file not found
        return {
            'content': """
# Default Technical Principles
- Library-First: Every feature starts as a standalone library
- CLI Interface: All libraries expose CLI functionality
- Test-First: TDD is mandatory (tests before implementation)
- Integration Testing: Focus on realistic test environments
- Simplicity: Start simple, add complexity only when justified
- Framework Trust: Use framework features directly
""",
            'path': 'default'
        }
    
    def analyze_input(self, user_input: str, context: Dict = None) -> AgentResponse:
        """Analyze user input from technical perspective"""
        
        technical_prompt = self._build_technical_analysis_prompt(user_input)
        if not self.check_budget_for_operation(technical_prompt):
            raise BudgetExceededException(
                self.token_tracker.get_total_cost(),
                self.token_tracker.session_budget
            )
        
        response = self._call_ai_api(technical_prompt, {
            'operation': 'technical_analysis',
            'agent_role': 'technical_architect'
        })
        
        return response
    
    def review_other_agent_response(self, 
                                   other_response: AgentResponse,
                                   original_input: str) -> AgentResponse:
        """Review another agent's response from technical perspective"""
        
        review_prompt = self._build_technical_review_prompt(other_response, original_input)
        
        if not self.check_budget_for_operation(review_prompt):
            return AgentResponse(
                agent_name=self.role,
                operation='budget_limited_review',
                content=f"Budget constraints limit detailed technical review of {other_response.agent_name} response.",
                metadata={'budget_limited': True},
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                timestamp="",
                confidence=0.5
            )
        
        response = self._call_ai_api(review_prompt, {
            'operation': 'technical_review',
            'reviewing': other_response.agent_name
        })
        
        return response
    
    def build_consensus_contribution(self, 
                                   all_responses: List[AgentResponse],
                                   original_input: str) -> AgentResponse:
        """Contribute to consensus building from technical perspective"""
        
        consensus_prompt = self._build_technical_consensus_prompt(all_responses, original_input)
        
        response = self._call_ai_api(consensus_prompt, {
            'operation': 'technical_consensus',
            'participant_count': len(all_responses)
        })
        
        return response
    
    def validate_constitutional_compliance(self, implementation_plan: str) -> Dict:
        """Validate implementation plan against constitution"""
        
        compliance_issues = []
        recommendations = []
        
        constitution_content = self.constitution['content'].lower()
        plan_lower = implementation_plan.lower()
        
        # Check Library-First principle
        if 'library' not in plan_lower:
            compliance_issues.append("Library-First: Implementation should start as standalone library")
            recommendations.append("Restructure as library with clear API boundaries")
        
        # Check CLI Interface requirement
        if 'cli' not in plan_lower and 'command line' not in plan_lower:
            compliance_issues.append("CLI Interface: Missing command-line interface requirement")
            recommendations.append("Add CLI interface for library functionality")
        
        # Check Test-First approach
        if 'test' not in plan_lower or 'tdd' not in plan_lower:
            compliance_issues.append("Test-First: Missing test-driven development approach")
            recommendations.append("Implement strict TDD: tests before implementation")
        
        # Check for over-engineering indicators
        complexity_indicators = ['framework', 'pattern', 'architecture', 'abstraction']
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in plan_lower)
        
        if complexity_count > 3:
            compliance_issues.append("Simplicity: Potential over-engineering detected")
            recommendations.append("Simplify approach, justify any added complexity")
        
        return {
            'compliant': len(compliance_issues) == 0,
            'issues': compliance_issues,
            'recommendations': recommendations,
            'constitution_path': self.constitution['path']
        }
    
    def _build_technical_analysis_prompt(self, user_input: str) -> str:
        """Build prompt for technical analysis"""
        return f"""
As a Technical Architect, analyze this feature request for implementation feasibility:

USER INPUT: {user_input}

CONSTITUTION PRINCIPLES:
{self.constitution['content'][:1000]}...

Please provide analysis in this structure:

## TECHNICAL FEASIBILITY ASSESSMENT
- Implementation complexity: [Low/Medium/High]
- Estimated development effort: [timeframe]
- Key technical challenges identified
- Dependency requirements

## ARCHITECTURE APPROACH
- Recommended system architecture
- Component breakdown and responsibilities  
- Data flow and integration points
- API design considerations

## CONSTITUTIONAL COMPLIANCE CHECK
- Library-First: How will this be structured as a standalone library?
- CLI Interface: What CLI commands will be exposed?
- Test-First: What testing strategy is required?
- Integration Focus: What integration testing is needed?

## IMPLEMENTATION PHASES
Break down into logical implementation phases:
1. Phase 1: [Core library/foundation]
2. Phase 2: [Key features]
3. Phase 3: [Integration/polish]

## TECHNICAL CONSTRAINTS & RISKS
- Performance considerations
- Scalability limitations
- Security implications
- Technical risks and mitigation strategies

## TECHNOLOGY RECOMMENDATIONS
- Suggested tech stack (if relevant)
- Framework/library choices with rationale
- Infrastructure requirements

Focus on technical feasibility while adhering to constitutional principles.
Be specific about implementation approach and identify potential blockers.
"""
    
    def _build_technical_review_prompt(self, other_response: AgentResponse, original_input: str) -> str:
        """Build prompt for technical review"""
        return f"""
As a Technical Architect, review this {other_response.agent_name} response for technical implementation implications:

ORIGINAL REQUEST: {original_input}

{other_response.agent_name.upper()} RESPONSE:
{other_response.content}

CONSTITUTION PRINCIPLES TO CHECK:
{self.constitution['content'][:800]}...

Please provide technical review focusing on:

## IMPLEMENTATION FEASIBILITY
- Are the proposed requirements technically implementable?
- What technical complexity is implied by these requirements?
- Are there any technically impossible or impractical elements?

## ARCHITECTURAL IMPLICATIONS
- What system architecture would be needed?
- How would this integrate with existing systems?
- What are the key technical design decisions required?

## CONSTITUTIONAL COMPLIANCE
- Does this approach align with Library-First principle?
- How would CLI interface be implemented?
- What testing approach would be required?
- Any over-engineering concerns?

## TECHNICAL GAPS
- What technical details are missing from requirements?
- What implementation decisions need clarification?
- What technical constraints should be added?

## RECOMMENDATIONS
- Technical approach suggestions
- Architecture recommendations
- Implementation strategy advice
- Risk mitigation suggestions

## EFFORT ESTIMATION
- Development complexity assessment
- Estimated implementation effort
- Key technical milestones

Provide constructive technical feedback focused on implementability and architectural soundness.
"""
    
    def _build_technical_consensus_prompt(self, all_responses: List[AgentResponse], original_input: str) -> str:
        """Build prompt for technical consensus contribution"""
        
        responses_summary = "\n\n".join([
            f"=== {resp.agent_name.upper()} PERSPECTIVE ===\n{resp.content}"
            for resp in all_responses
        ])
        
        return f"""
As a Technical Architect, synthesize all perspectives into a technically sound implementation plan:

ORIGINAL REQUEST: {original_input}

ALL AGENT RESPONSES:
{responses_summary}

CONSTITUTION REQUIREMENTS:
{self.constitution['content'][:600]}...

Provide technical consensus focusing on:

## UNIFIED TECHNICAL APPROACH
Synthesize the best technical insights from all agents:
- Core architecture decisions
- Implementation strategy that addresses all concerns
- Technology choices with rationale

## CONSTITUTIONAL IMPLEMENTATION
How will constitutional principles be implemented:
- Library structure and boundaries
- CLI interface design
- Test-first development approach
- Integration testing strategy

## CONSENSUS TECHNICAL REQUIREMENTS
Technical requirements that satisfy all agent perspectives:
- Functional technical specifications
- Non-functional requirements (performance, scalability, security)
- Integration requirements
- Testing requirements

## IMPLEMENTATION ROADMAP
Phased implementation plan:
- Phase 1: Foundation (library structure, core functionality)
- Phase 2: Feature completion (full requirements implementation)
- Phase 3: Integration & optimization (testing, performance)

## CONFLICT RESOLUTION
Technical approach to resolve agent disagreements:
- Trade-off analysis and recommendations
- Technical feasibility considerations
- Risk vs benefit assessment

## FINAL TECHNICAL SPECIFICATION
Definitive technical specification incorporating all valid concerns:
- Architecture overview
- Key components and interfaces
- Implementation approach
- Testing strategy
- Success criteria

Focus on creating a technically sound plan that satisfies all stakeholder concerns while maintaining constitutional compliance.
"""