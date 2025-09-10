#!/usr/bin/env python3
"""
QA Agent for collaborative specification generation.
Focuses on testability, quality metrics, and validation scenarios.
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentResponse, AgentCapability


class QAAgent(BaseAgent):
    """AI agent that thinks like a QA Engineer/Test Architect"""
    
    def _load_configuration(self):
        """Load QA-specific configuration"""
        self.focus_areas = [
            'test_scenario_design',
            'edge_case_identification',
            'quality_metrics_definition', 
            'validation_strategies',
            'acceptance_testing',
            'test_automation_strategy'
        ]
        
        self.capabilities = [
            AgentCapability(
                name="design_test_scenarios",
                description="Create comprehensive test scenarios and edge cases",
                input_format="Feature requirements and acceptance criteria",
                output_format="Test scenarios, test cases, and validation procedures",
                estimated_tokens=1200
            ),
            AgentCapability(
                name="identify_quality_risks",
                description="Identify quality risks and testing challenges",
                input_format="Implementation approach and requirements",
                output_format="Quality risk assessment and mitigation strategies",
                estimated_tokens=800
            ),
            AgentCapability(
                name="define_acceptance_criteria",
                description="Define testable acceptance criteria and success metrics",
                input_format="User stories and business requirements",
                output_format="Detailed acceptance criteria and quality gates",
                estimated_tokens=600
            )
        ]
    
    def analyze_input(self, user_input: str, context: Dict = None) -> AgentResponse:
        """Analyze user input from QA perspective"""
        
        qa_prompt = self._build_qa_analysis_prompt(user_input)
        if not self.check_budget_for_operation(qa_prompt):
            raise BudgetExceededException(
                self.token_tracker.get_total_cost(),
                self.token_tracker.session_budget
            )
        
        response = self._call_ai_api(qa_prompt, {
            'operation': 'qa_analysis',
            'agent_role': 'qa_engineer'
        })
        
        return response
    
    def review_other_agent_response(self, 
                                   other_response: AgentResponse,
                                   original_input: str) -> AgentResponse:
        """Review another agent's response from QA perspective"""
        
        review_prompt = self._build_qa_review_prompt(other_response, original_input)
        
        if not self.check_budget_for_operation(review_prompt):
            return AgentResponse(
                agent_name=self.role,
                operation='budget_limited_review',
                content=f"Budget constraints limit QA review of {other_response.agent_name} response.",
                metadata={'budget_limited': True},
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                timestamp="",
                confidence=0.5
            )
        
        response = self._call_ai_api(review_prompt, {
            'operation': 'qa_review',
            'reviewing': other_response.agent_name
        })
        
        return response
    
    def build_consensus_contribution(self, 
                                   all_responses: List[AgentResponse],
                                   original_input: str) -> AgentResponse:
        """Contribute to consensus building from QA perspective"""
        
        consensus_prompt = self._build_qa_consensus_prompt(all_responses, original_input)
        
        response = self._call_ai_api(consensus_prompt, {
            'operation': 'qa_consensus',
            'participant_count': len(all_responses)
        })
        
        return response
    
    def _build_qa_analysis_prompt(self, user_input: str) -> str:
        """Build prompt for QA analysis"""
        return f"""
As a QA Engineer, analyze this feature request for testability and quality considerations:

USER INPUT: {user_input}

Please provide QA analysis in this structure:

## TESTABILITY ASSESSMENT
- How can this feature be effectively tested?
- What are the key test scenarios required?
- What testing challenges or complexities exist?
- What test data or test environments are needed?

## TEST SCENARIO DESIGN
Create comprehensive test scenarios:
- Happy path scenarios (primary user flows)
- Alternative path scenarios (edge cases)
- Error path scenarios (failure conditions)
- Boundary condition tests

## EDGE CASE IDENTIFICATION
- What unusual or extreme conditions should be tested?
- What input validation scenarios are needed?
- What concurrent user scenarios should be considered?
- What system limit scenarios should be tested?

## QUALITY METRICS & SUCCESS CRITERIA
- How will feature quality be measured?
- What are the performance acceptance criteria?
- What are the reliability requirements?
- What user experience quality standards apply?

## ACCEPTANCE TESTING STRATEGY
- What user acceptance testing is required?
- How should stakeholder validation occur?
- What demonstration scenarios are needed?
- What sign-off criteria should be established?

## TEST AUTOMATION APPROACH
- What tests should be automated vs manual?
- What test frameworks or tools are appropriate?
- What CI/CD integration is needed for testing?
- What regression testing coverage is required?

## QUALITY RISK ASSESSMENT
- What are the highest quality risks?
- What could go wrong with this feature?
- What are the consequences of quality failures?
- What risk mitigation testing is needed?

## TESTING EFFORT ESTIMATION
- Testing complexity assessment (Low/Medium/High)
- Estimated testing effort and timeline
- Required testing resources and skills
- Critical testing milestones

Focus on ensuring the feature can be thoroughly validated and meets quality standards.
Be specific about test scenarios and quality criteria.
"""
    
    def _build_qa_review_prompt(self, other_response: AgentResponse, original_input: str) -> str:
        """Build prompt for QA review"""
        return f"""
As a QA Engineer, review this {other_response.agent_name} response for testability and quality implications:

ORIGINAL REQUEST: {original_input}

{other_response.agent_name.upper()} RESPONSE:
{other_response.content}

Please provide QA review focusing on:

## TESTABILITY EVALUATION
- How testable are the proposed requirements/approach?
- What testing challenges does this approach create?
- Are requirements specific enough to create tests?
- What acceptance criteria need clarification?

## MISSING TEST SCENARIOS
- What test scenarios are not covered by this response?
- What edge cases or error conditions are overlooked?
- What integration testing scenarios are needed?
- What performance or load testing considerations exist?

## QUALITY CONCERNS
- What quality risks does this approach introduce?
- Are there reliability or stability concerns?
- What user experience quality issues might arise?
- Are there maintainability or supportability concerns?

## TEST STRATEGY GAPS
- What testing approach is needed for this solution?
- What types of testing are missing from consideration?
- How should test automation be approached?
- What test data and environments are required?

## VALIDATION REQUIREMENTS
- How should this feature be validated with users?
- What demonstration or acceptance criteria are needed?
- What success metrics should be established?
- How will quality be measured and monitored?

## RECOMMENDATIONS
- Testing strategies to enhance the approach
- Quality improvements to consider
- Specific test scenarios to add
- Quality gates and checkpoints to establish

## CONSENSUS OPPORTUNITIES
- How testing requirements align with other concerns
- Where quality considerations support business goals
- How testing can validate technical and business requirements

Provide constructive QA feedback that ensures the solution can be properly validated.
Focus on building quality and testability into the approach from the start.
"""
    
    def _build_qa_consensus_prompt(self, all_responses: List[AgentResponse], original_input: str) -> str:
        """Build prompt for QA consensus contribution"""
        
        responses_summary = "\n\n".join([
            f"=== {resp.agent_name.upper()} PERSPECTIVE ===\n{resp.content}"
            for resp in all_responses
        ])
        
        return f"""
As a QA Engineer, synthesize testing and quality requirements that validate all stakeholder perspectives:

ORIGINAL REQUEST: {original_input}

ALL AGENT RESPONSES:
{responses_summary}

Provide QA consensus focusing on:

## INTEGRATED TESTING STRATEGY
Testing approach that validates all stakeholder concerns:
- Business requirements validation through acceptance testing
- Technical implementation validation through integration/system testing
- Security requirements validation through security testing
- Quality requirements validation through quality assurance processes

## COMPREHENSIVE TEST SCENARIOS
Unified test scenarios addressing all perspectives:
- User story validation tests (business perspective)
- Technical functionality tests (technical perspective)
- Security validation tests (security perspective)
- Quality attribute tests (performance, reliability, usability)

## QUALITY GATES INTEGRATION
Quality checkpoints that ensure all requirements are met:
- Requirements validation gates
- Implementation quality gates
- Security validation gates
- User acceptance gates

## CONSENSUS ACCEPTANCE CRITERIA
Testable acceptance criteria that satisfy all stakeholders:
- Functional acceptance criteria (what the system does)
- Non-functional acceptance criteria (how well it does it)
- Business acceptance criteria (value delivered)
- Technical acceptance criteria (implementation quality)

## CONFLICT RESOLUTION TESTING
Testing approach to validate conflict resolutions:
- Test scenarios that validate trade-off decisions
- Quality metrics to evaluate different approaches
- Acceptance criteria for compromise solutions
- Risk validation through targeted testing

## FINAL TESTING SPECIFICATION
Comprehensive testing approach incorporating all perspectives:
- Test strategy and approach
- Test scenarios and test cases
- Quality metrics and success criteria
- Testing timeline and milestones
- Test automation strategy
- Risk-based testing priorities

## QUALITY SUCCESS DEFINITION
Clear definition of quality success that addresses all stakeholder needs:
- Business success criteria
- Technical quality criteria
- Security validation criteria
- User experience quality criteria

Focus on ensuring the final solution can be thoroughly validated against all stakeholder requirements.
Create testing strategy that gives confidence in the complete solution.
"""