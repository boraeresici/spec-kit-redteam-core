#!/usr/bin/env python3
"""
Product Manager Agent for collaborative specification generation.
Focuses on user value, business requirements, and acceptance criteria.
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentResponse, AgentCapability


class ProductManagerAgent(BaseAgent):
    """AI agent that thinks like a Product Manager"""
    
    def _load_configuration(self):
        """Load PM-specific configuration"""
        self.focus_areas = [
            'user_value_proposition',
            'business_requirements', 
            'user_journey_mapping',
            'acceptance_criteria',
            'stakeholder_needs',
            'success_metrics'
        ]
        
        self.capabilities = [
            AgentCapability(
                name="analyze_user_requirements",
                description="Extract and clarify user requirements from descriptions",
                input_format="Natural language feature description",
                output_format="Structured user stories and acceptance criteria",
                estimated_tokens=1500
            ),
            AgentCapability(
                name="define_user_scenarios",
                description="Create comprehensive user scenarios and edge cases",
                input_format="Feature requirements",
                output_format="Given-When-Then scenarios",
                estimated_tokens=1200
            ),
            AgentCapability(
                name="identify_business_value",
                description="Identify and articulate business value and success metrics",
                input_format="Feature description and context",
                output_format="Business value statement and KPIs",
                estimated_tokens=800
            )
        ]
    
    def analyze_input(self, user_input: str, context: Dict = None) -> AgentResponse:
        """Analyze user input from PM perspective"""
        
        # Check budget before proceeding
        pm_prompt = self._build_analysis_prompt(user_input)
        if not self.check_budget_for_operation(pm_prompt):
            raise BudgetExceededException(
                self.token_tracker.get_total_cost(),
                self.token_tracker.session_budget
            )
        
        # Generate PM analysis
        response = self._call_ai_api(pm_prompt, {
            'operation': 'initial_analysis',
            'agent_role': 'product_manager'
        })
        
        return response
    
    def review_other_agent_response(self, 
                                   other_response: AgentResponse,
                                   original_input: str) -> AgentResponse:
        """Review another agent's response from PM perspective"""
        
        review_prompt = self._build_review_prompt(other_response, original_input)
        
        if not self.check_budget_for_operation(review_prompt):
            # Return minimal review if budget is tight
            return AgentResponse(
                agent_name=self.role,
                operation='budget_limited_review',
                content=f"Budget constraints limit detailed review of {other_response.agent_name} response.",
                metadata={'budget_limited': True},
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                timestamp="",
                confidence=0.5
            )
        
        response = self._call_ai_api(review_prompt, {
            'operation': 'peer_review',
            'reviewing': other_response.agent_name
        })
        
        return response
    
    def build_consensus_contribution(self, 
                                   all_responses: List[AgentResponse],
                                   original_input: str) -> AgentResponse:
        """Contribute to consensus building from PM perspective"""
        
        consensus_prompt = self._build_consensus_prompt(all_responses, original_input)
        
        response = self._call_ai_api(consensus_prompt, {
            'operation': 'consensus_building',
            'participant_count': len(all_responses)
        })
        
        return response
    
    def _build_analysis_prompt(self, user_input: str) -> str:
        """Build prompt for initial PM analysis"""
        return f"""
As a Product Manager, analyze this feature request and provide structured output:

USER INPUT: {user_input}

Please provide analysis in the following structure:

## USER VALUE ANALYSIS
- What user problem does this solve?
- What is the core value proposition?
- Who are the target users/personas?

## BUSINESS REQUIREMENTS
- What business objectives does this support?
- What are the key success metrics/KPIs?
- What constraints or dependencies exist?

## USER SCENARIOS
Create 2-3 primary user scenarios in Given-When-Then format:
- Given [context], When [action], Then [outcome]

## ACCEPTANCE CRITERIA  
List specific, testable acceptance criteria:
- Must have: [critical requirements]
- Should have: [important requirements]
- Could have: [nice to have requirements]

## CLARIFICATION NEEDS
Mark any unclear aspects with [NEEDS CLARIFICATION: specific question]

## SCOPE BOUNDARIES
- What is explicitly IN scope?
- What is explicitly OUT of scope?

Focus on WHAT users need and WHY, not HOW to implement.
Be specific and testable. Mark ambiguities clearly.
"""
    
    def _build_review_prompt(self, other_response: AgentResponse, original_input: str) -> str:
        """Build prompt for reviewing another agent's response"""
        return f"""
As a Product Manager, review this {other_response.agent_name} agent's response to ensure it aligns with user needs and business value:

ORIGINAL REQUEST: {original_input}

{other_response.agent_name.upper()} RESPONSE:
{other_response.content}

Please provide feedback focusing on:

## ALIGNMENT CHECK
- Does this response address the core user needs identified?
- Are there business requirements that are missed or misunderstood?
- Does the approach support the intended user value proposition?

## MISSING ELEMENTS
- What user scenarios or edge cases are not addressed?
- Are there acceptance criteria that need refinement?
- What stakeholder needs might be overlooked?

## RECOMMENDATIONS
- Specific suggestions for improvement
- Additional considerations from PM perspective
- Risk mitigation suggestions

## CONSENSUS POINTS
- Areas where you agree with the {other_response.agent_name} perspective
- Shared understanding points
- Compatible approaches

Keep feedback constructive and focused on user/business value alignment.
"""
    
    def _build_consensus_prompt(self, all_responses: List[AgentResponse], original_input: str) -> str:
        """Build prompt for consensus contribution"""
        
        responses_summary = "\n\n".join([
            f"=== {resp.agent_name.upper()} PERSPECTIVE ===\n{resp.content}"
            for resp in all_responses
        ])
        
        return f"""
As a Product Manager, help build consensus from all agent perspectives:

ORIGINAL REQUEST: {original_input}

ALL AGENT RESPONSES:
{responses_summary}

Please provide a PM-driven consensus contribution:

## SYNTHESIZED USER STORIES
Combine the best insights from all agents into clear user stories:
- As a [user type], I want [capability], so that [benefit]

## UNIFIED ACCEPTANCE CRITERIA
Merge all valid acceptance criteria into a prioritized list:
- Critical (Must Have): [requirements that must be met]
- Important (Should Have): [requirements that add significant value]
- Optional (Could Have): [nice-to-have requirements]

## CONSENSUS AREAS
- What do all agents agree on?
- What are the shared priorities?
- What approach elements are universally supported?

## CONFLICT RESOLUTION
- Where do agents disagree?
- What are the trade-offs involved?
- Recommended resolution from business value perspective

## FINAL REQUIREMENTS SYNTHESIS
Provide the definitive requirements that incorporate the best insights from all perspectives while maintaining focus on user value and business objectives.

Prioritize user value and business feasibility in conflict resolution.
"""