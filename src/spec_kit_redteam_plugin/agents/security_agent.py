#!/usr/bin/env python3
"""
Security Agent for collaborative specification generation.
Focuses on threat modeling, compliance, and security requirements.
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentResponse, AgentCapability


class SecurityAgent(BaseAgent):
    """AI agent that thinks like a Security Architect"""
    
    def _load_configuration(self):
        """Load Security-specific configuration"""
        self.focus_areas = [
            'threat_modeling',
            'data_protection', 
            'authentication_authorization',
            'compliance_requirements',
            'security_testing',
            'vulnerability_assessment'
        ]
        
        self.capabilities = [
            AgentCapability(
                name="perform_threat_analysis",
                description="Identify security threats and attack vectors",
                input_format="Feature requirements and architecture",
                output_format="Threat model and security requirements",
                estimated_tokens=1500
            ),
            AgentCapability(
                name="assess_compliance_needs",
                description="Evaluate compliance requirements (GDPR, SOC2, etc.)",
                input_format="Feature description and data handling",
                output_format="Compliance assessment and requirements",
                estimated_tokens=1000
            ),
            AgentCapability(
                name="security_testing_strategy",
                description="Define security testing approach",
                input_format="Implementation plan",
                output_format="Security testing requirements and procedures",
                estimated_tokens=800
            )
        ]
    
    def analyze_input(self, user_input: str, context: Dict = None) -> AgentResponse:
        """Analyze user input from security perspective"""
        
        security_prompt = self._build_security_analysis_prompt(user_input)
        if not self.check_budget_for_operation(security_prompt):
            raise BudgetExceededException(
                self.token_tracker.get_total_cost(),
                self.token_tracker.session_budget
            )
        
        response = self._call_ai_api(security_prompt, {
            'operation': 'security_analysis',
            'agent_role': 'security_architect'
        })
        
        return response
    
    def review_other_agent_response(self, 
                                   other_response: AgentResponse,
                                   original_input: str) -> AgentResponse:
        """Review another agent's response from security perspective"""
        
        review_prompt = self._build_security_review_prompt(other_response, original_input)
        
        if not self.check_budget_for_operation(review_prompt):
            return AgentResponse(
                agent_name=self.role,
                operation='budget_limited_review',
                content=f"Budget constraints limit security review of {other_response.agent_name} response.",
                metadata={'budget_limited': True},
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                timestamp="",
                confidence=0.5
            )
        
        response = self._call_ai_api(review_prompt, {
            'operation': 'security_review',
            'reviewing': other_response.agent_name
        })
        
        return response
    
    def build_consensus_contribution(self, 
                                   all_responses: List[AgentResponse],
                                   original_input: str) -> AgentResponse:
        """Contribute to consensus building from security perspective"""
        
        consensus_prompt = self._build_security_consensus_prompt(all_responses, original_input)
        
        response = self._call_ai_api(consensus_prompt, {
            'operation': 'security_consensus',
            'participant_count': len(all_responses)
        })
        
        return response
    
    def _build_security_analysis_prompt(self, user_input: str) -> str:
        """Build prompt for security analysis"""
        return f"""
As a Security Architect, analyze this feature request for security implications:

USER INPUT: {user_input}

Please provide security analysis in this structure:

## THREAT MODEL ANALYSIS
- What are the key assets being protected?
- Who are the potential threat actors?
- What are the primary attack vectors?
- What is the impact of potential breaches?

## DATA SECURITY ASSESSMENT
- What types of data are involved (PII, credentials, business data)?
- How should data be classified (public, internal, confidential, restricted)?
- What data protection measures are required?
- Are there data residency or sovereignty requirements?

## AUTHENTICATION & AUTHORIZATION
- What authentication mechanisms are needed?
- What authorization controls should be implemented?
- How should user sessions be managed?
- Are there role-based access control requirements?

## COMPLIANCE CONSIDERATIONS  
- What compliance frameworks might apply (GDPR, HIPAA, SOC2, PCI)?
- What audit and logging requirements exist?
- Are there regulatory reporting needs?
- What data retention/deletion policies are required?

## SECURITY REQUIREMENTS
- Input validation and sanitization requirements
- Encryption requirements (in transit and at rest)
- Secure communication protocols needed
- Security headers and configurations

## SECURITY TESTING STRATEGY
- What security testing is required?
- What vulnerability assessments are needed?
- What penetration testing scope is appropriate?
- What security monitoring should be implemented?

## RISK ASSESSMENT
- High/Medium/Low risk classification
- Key security risks and their likelihood/impact
- Risk mitigation strategies
- Residual risk acceptance criteria

Focus on identifying security requirements that must be built into the feature from the start.
Be specific about security controls and testing needs.
"""
    
    def _build_security_review_prompt(self, other_response: AgentResponse, original_input: str) -> str:
        """Build prompt for security review"""
        return f"""
As a Security Architect, review this {other_response.agent_name} response for security considerations:

ORIGINAL REQUEST: {original_input}

{other_response.agent_name.upper()} RESPONSE:
{other_response.content}

Please provide security review focusing on:

## SECURITY GAP ANALYSIS
- What security considerations are missing from this response?
- Are there unaddressed threat vectors?
- What data protection measures are overlooked?
- Are authentication/authorization requirements adequate?

## THREAT VECTOR ASSESSMENT
- What new attack surfaces does this approach create?
- How might an attacker exploit the proposed functionality?
- What are the security implications of the suggested architecture?
- Are there privilege escalation risks?

## COMPLIANCE IMPACT
- How does this approach affect regulatory compliance?
- What audit trail requirements are needed?
- Are there data handling compliance issues?
- What privacy considerations are missing?

## SECURITY INTEGRATION
- How can security controls be integrated into this approach?
- What security testing should be added?
- How should security monitoring be implemented?
- What incident response considerations exist?

## RECOMMENDATIONS
- Critical security requirements to add
- Security architecture modifications needed
- Additional security testing requirements
- Risk mitigation strategies

## CONSENSUS OPPORTUNITIES  
- Where security requirements align with other concerns
- How security can enhance the overall solution
- Shared security and functionality benefits

Provide constructive security feedback that enhances rather than blocks the solution.
Focus on building security in, not adding it as an afterthought.
"""
    
    def _build_security_consensus_prompt(self, all_responses: List[AgentResponse], original_input: str) -> str:
        """Build prompt for security consensus contribution"""
        
        responses_summary = "\n\n".join([
            f"=== {resp.agent_name.upper()} PERSPECTIVE ===\n{resp.content}"
            for resp in all_responses
        ])
        
        return f"""
As a Security Architect, synthesize security requirements that integrate with all other perspectives:

ORIGINAL REQUEST: {original_input}

ALL AGENT RESPONSES:
{responses_summary}

Provide security consensus focusing on:

## INTEGRATED SECURITY APPROACH
Security strategy that supports all stakeholder needs:
- Security controls that enable rather than hinder functionality
- Threat mitigation that aligns with business requirements
- Security testing integrated with overall quality strategy

## SECURITY REQUIREMENTS SYNTHESIS
Consolidated security requirements addressing all concerns:
- Authentication and authorization requirements
- Data protection and privacy requirements
- Secure communication and integration requirements
- Security monitoring and incident response requirements

## SECURITY-BY-DESIGN INTEGRATION
How security integrates with proposed technical approach:
- Security controls built into architecture
- Secure development practices integration
- Security testing as part of overall test strategy
- Security monitoring integrated with system observability

## CONSENSUS SECURITY POSTURE
Balanced security approach that satisfies all stakeholders:
- Risk-appropriate security controls
- Compliance requirements integrated with business needs
- Security that enhances user experience rather than degrading it
- Cost-effective security measures

## CONFLICT RESOLUTION
Security perspective on resolving disagreements:
- Security implications of different approaches
- Risk-based decision making framework
- Security trade-offs and recommendations
- Non-negotiable security requirements vs. flexible implementations

## FINAL SECURITY SPECIFICATION
Definitive security requirements that integrate with all perspectives:
- Core security controls (must-have)
- Enhanced security features (should-have)
- Advanced security capabilities (could-have)
- Security testing and validation approach
- Security success criteria and metrics

Ensure security requirements enhance the overall solution while maintaining necessary protection.
Focus on pragmatic security that enables business value.
"""