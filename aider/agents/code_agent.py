"""
CodeAgent: Specialized agent for code generation, modification, and review.

This agent handles all code-related operations in the Aider Multi-Agent Hive Architecture:
- Code generation from natural language descriptions
- Code modification and refactoring
- Code review and quality assessment
- Debugging and error resolution
- Syntax validation and testing
- Multi-language support
"""

import asyncio
import ast
import json
import logging
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from .base_agent import BaseAgent, AgentMessage, MessagePriority, AgentState, AgentCapability
from ..models.model_manager import generate_code, analyze_code, ComplexityLevel


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class CodeOperation(Enum):
    """Types of code operations."""
    GENERATE = "generate"
    MODIFY = "modify"
    REVIEW = "review"
    DEBUG = "debug"
    REFACTOR = "refactor"
    VALIDATE = "validate"
    TEST = "test"
    OPTIMIZE = "optimize"
    DOCUMENT = "document"


class CodeQuality(Enum):
    """Code quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class CodeAnalysis:
    """Results of code analysis."""
    language: CodeLanguage
    lines_of_code: int
    complexity_score: float
    quality_score: float
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    metrics: Dict[str, Any]
    validation_passed: bool
    test_coverage: Optional[float] = None


@dataclass
class CodeGenerationRequest:
    """Request for code generation."""
    description: str
    language: CodeLanguage
    context: Optional[Dict[str, Any]] = None
    requirements: List[str] = None
    style_guide: Optional[str] = None
    existing_code: Optional[str] = None
    target_file: Optional[str] = None


@dataclass
class CodeGenerationResult:
    """Result of code generation."""
    generated_code: str
    analysis: CodeAnalysis
    explanation: str
    test_code: Optional[str] = None
    documentation: Optional[str] = None
    confidence_score: float = 0.0


class CodeAgent(BaseAgent):
    """
    Specialized agent for code-related operations.

    Responsibilities:
    - Generate code from natural language descriptions
    - Modify and refactor existing code
    - Review code for quality and potential issues
    - Debug code and suggest fixes
    - Validate code syntax and semantics
    - Support multiple programming languages
    """

    def __init__(
        self,
        agent_id: str = "code_agent",
        config: Optional[Dict[str, Any]] = None,
        message_bus=None,
    ):
        """Initialize the code agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type="code",
            config=config,
            message_bus=message_bus,
        )

        # Code agent specific configuration
        self.supported_languages = set(CodeLanguage)
        self.max_file_size = self.config.get('max_file_size', 1024 * 1024)  # 1MB
        self.generation_timeout = self.config.get('generation_timeout', 120.0)
        self.enable_testing = self.config.get('enable_testing', True)
        self.enable_validation = self.config.get('enable_validation', True)

        # Language-specific settings
        self.language_extensions = {
            CodeLanguage.PYTHON: ['.py', '.pyw'],
            CodeLanguage.JAVASCRIPT: ['.js', '.mjs'],
            CodeLanguage.TYPESCRIPT: ['.ts', '.tsx'],
            CodeLanguage.JAVA: ['.java'],
            CodeLanguage.CPP: ['.cpp', '.cxx', '.cc'],
            CodeLanguage.C: ['.c', '.h'],
            CodeLanguage.CSHARP: ['.cs'],
            CodeLanguage.GO: ['.go'],
            CodeLanguage.RUST: ['.rs'],
            CodeLanguage.PHP: ['.php'],
            CodeLanguage.RUBY: ['.rb'],
            CodeLanguage.KOTLIN: ['.kt'],
            CodeLanguage.SWIFT: ['.swift'],
            CodeLanguage.SHELL: ['.sh', '.bash', '.zsh'],
            CodeLanguage.SQL: ['.sql'],
            CodeLanguage.HTML: ['.html', '.htm'],
            CodeLanguage.CSS: ['.css', '.scss', '.sass'],
            CodeLanguage.YAML: ['.yaml', '.yml'],
            CodeLanguage.JSON: ['.json'],
            CodeLanguage.XML: ['.xml'],
            CodeLanguage.MARKDOWN: ['.md', '.markdown'],
        }

        # Code quality metrics
        self.quality_thresholds = {
            'complexity_max': 10,
            'function_length_max': 50,
            'line_length_max': 120,
            'maintainability_min': 70
        }

        # Performance metrics
        self.generation_metrics = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_generation_time': 0.0,
            'language_distribution': {},
            'quality_distribution': {}
        }

        # Template patterns for different operations
        self.code_templates = {
            CodeLanguage.PYTHON: {
                'function': '''def {function_name}({parameters}):
    """
    {description}

    Args:
        {args_docs}

    Returns:
        {return_docs}
    """
    {body}''',
                'class': '''class {class_name}:
    """
    {description}
    """

    def __init__(self{init_params}):
        """Initialize the {class_name}."""
        {init_body}

    {methods}''',
                'test': '''import unittest
from {module} import {class_or_function}


class Test{name}(unittest.TestCase):
    """Test cases for {name}."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    {test_methods}


if __name__ == '__main__':
    unittest.main()'''
            }
        }

    async def initialize(self) -> bool:
        """Initialize the code agent."""
        try:
            await super().initialize()

            # Register code-specific message handlers
            self.register_message_handler('generate_code', self._handle_generate_code)
            self.register_message_handler('modify_code', self._handle_modify_code)
            self.register_message_handler('review_code', self._handle_review_code)
            self.register_message_handler('debug_code', self._handle_debug_code)
            self.register_message_handler('validate_code', self._handle_validate_code)
            self.register_message_handler('task_request', self._handle_task_request)

            # Validate environment
            await self._validate_environment()

            self.logger.info("CodeAgent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize CodeAgent: {e}")
            return False

    async def _validate_environment(self) -> None:
        """
        Validate the environment for code operations.

        Checks for required tools and configurations.
        """
        try:
            # Check if we can create temporary files
            with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmp:
                tmp.write("# Test file\n")
                tmp.flush()

            # Validate code analysis capabilities
            self.code_metrics = {
                'environment_validated': True,
                'temp_file_access': True,
                'analysis_ready': True
            }

            self.logger.debug("CodeAgent environment validation completed successfully")

        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            raise

    async def generate_code(
        self,
        request: CodeGenerationRequest
    ) -> CodeGenerationResult:
        """
        Generate code based on the provided request.

        Args:
            request: Code generation request with description and requirements

        Returns:
            CodeGenerationResult with generated code and analysis
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        self.logger.info(
            "Starting code generation",
            request_id=request_id,
            language=request.language.value,
            description=request.description[:100] + "..." if len(request.description) > 100 else request.description
        )

        try:
            # Analyze the request
            analysis_context = await self._analyze_generation_request(request)

            # Generate the code
            generated_code = await self._generate_code_content(request, analysis_context)

            # Analyze the generated code
            code_analysis = await self._analyze_code(generated_code, request.language)

            # Generate tests if enabled
            test_code = None
            if self.enable_testing and request.language in [CodeLanguage.PYTHON, CodeLanguage.JAVASCRIPT]:
                test_code = await self._generate_test_code(generated_code, request)

            # Generate documentation
            documentation = await self._generate_documentation(generated_code, request)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(code_analysis, request)

            # Create result
            result = CodeGenerationResult(
                generated_code=generated_code,
                analysis=code_analysis,
                explanation=self._generate_explanation(generated_code, request, analysis_context),
                test_code=test_code,
                documentation=documentation,
                confidence_score=confidence_score
            )

            # Update metrics
            generation_time = time.time() - start_time
            self._update_generation_metrics(request.language, generation_time, True)

            self.logger.info(
                "Code generation completed",
                request_id=request_id,
                generation_time=generation_time,
                confidence_score=confidence_score,
                quality_score=code_analysis.quality_score
            )

            return result

        except Exception as e:
            self.logger.error(
                "Code generation failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            self._update_generation_metrics(request.language, time.time() - start_time, False)
            raise

    async def modify_code(
        self,
        existing_code: str,
        modifications: str,
        language: CodeLanguage,
        context: Optional[Dict[str, Any]] = None
    ) -> CodeGenerationResult:
        """
        Modify existing code based on requirements.

        Args:
            existing_code: The code to modify
            modifications: Description of required modifications
            language: Programming language of the code
            context: Optional context information

        Returns:
            CodeGenerationResult with modified code and analysis
        """
        self.logger.info("Starting code modification", language=language.value)

        try:
            # Analyze existing code
            existing_analysis = await self._analyze_code(existing_code, language)

            # Create modification request
            modification_request = CodeGenerationRequest(
                description=f"Modify the following code: {modifications}",
                language=language,
                context=context,
                existing_code=existing_code
            )

            # Generate modified code
            modified_code = await self._modify_code_content(existing_code, modifications, language, context)

            # Analyze modified code
            modified_analysis = await self._analyze_code(modified_code, language)

            # Generate explanation of changes
            explanation = await self._generate_modification_explanation(
                existing_code, modified_code, modifications, existing_analysis, modified_analysis
            )

            result = CodeGenerationResult(
                generated_code=modified_code,
                analysis=modified_analysis,
                explanation=explanation,
                confidence_score=self._calculate_modification_confidence(existing_analysis, modified_analysis)
            )

            self.logger.info("Code modification completed")
            return result

        except Exception as e:
            self.logger.error(f"Code modification failed: {e}", exc_info=True)
            raise

    async def review_code(
        self,
        code: str,
        language: CodeLanguage,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Review code for quality, style, and potential issues.

        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus on during review

        Returns:
            Dictionary with review results and recommendations
        """
        self.logger.info("Starting code review", language=language.value)

        try:
            # Analyze code quality
            analysis = await self._analyze_code(code, language)

            # Perform focused reviews
            review_results = {
                'overall_quality': analysis.quality_score,
                'complexity_analysis': await self._analyze_complexity(code, language),
                'style_analysis': await self._analyze_style(code, language),
                'security_analysis': await self._analyze_security(code, language),
                'performance_analysis': await self._analyze_performance(code, language),
                'maintainability_analysis': await self._analyze_maintainability(code, language)
            }

            # Generate recommendations
            recommendations = await self._generate_review_recommendations(analysis, review_results, focus_areas)

            # Create comprehensive review report
            review_report = {
                'analysis': analysis,
                'detailed_reviews': review_results,
                'recommendations': recommendations,
                'action_items': self._prioritize_action_items(analysis.issues),
                'quality_grade': self._assign_quality_grade(analysis.quality_score),
                'review_summary': self._generate_review_summary(analysis, review_results)
            }

            self.logger.info("Code review completed", quality_score=analysis.quality_score)
            return review_report

        except Exception as e:
            self.logger.error(f"Code review failed: {e}", exc_info=True)
            raise

    async def debug_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        language: CodeLanguage = CodeLanguage.PYTHON,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Debug code and suggest fixes for issues.

        Args:
            code: Code to debug
            error_message: Optional error message to help with debugging
            language: Programming language
            context: Optional context information

        Returns:
            Dictionary with debugging results and suggested fixes
        """
        self.logger.info("Starting code debugging", language=language.value)

        try:
            # Analyze code for issues
            analysis = await self._analyze_code(code, language)

            # Identify specific errors
            syntax_errors = await self._check_syntax_errors(code, language)
            logic_errors = await self._identify_logic_errors(code, language, error_message)
            runtime_errors = await self._identify_runtime_errors(code, language, error_message)

            # Generate fixes
            suggested_fixes = await self._generate_debug_fixes(
                code, syntax_errors, logic_errors, runtime_errors, language
            )

            # Test fixes if possible
            tested_fixes = await self._test_debug_fixes(code, suggested_fixes, language)

            debug_report = {
                'original_analysis': analysis,
                'syntax_errors': syntax_errors,
                'logic_errors': logic_errors,
                'runtime_errors': runtime_errors,
                'suggested_fixes': tested_fixes,
                'confidence_scores': self._calculate_fix_confidence(tested_fixes),
                'debugging_summary': self._generate_debug_summary(syntax_errors, logic_errors, runtime_errors)
            }

            self.logger.info("Code debugging completed")
            return debug_report

        except Exception as e:
            self.logger.error(f"Code debugging failed: {e}", exc_info=True)
            raise

    async def validate_code(
        self,
        code: str,
        language: CodeLanguage,
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Validate code using AI-powered analysis and traditional validation.

        Args:
            code: Code to validate
            language: Programming language
            strict_mode: Enable strict validation rules

        Returns:
            Dictionary with validation results
        """
        self.logger.debug("Validating code", language=language.value, strict_mode=strict_mode)

        try:
            validation_results = {
                'syntax_valid': True,
                'structure_valid': True,
                'semantic_valid': True,
                'errors': [],
                'warnings': [],
                'suggestions': []
            }

            # AI-powered validation first
            try:
                ai_validation = await analyze_code(code, "validation")
                ai_results = self._parse_ai_validation(ai_validation, language)

                # Merge AI validation results
                validation_results['errors'].extend(ai_results.get('errors', []))
                validation_results['warnings'].extend(ai_results.get('warnings', []))
                validation_results['suggestions'].extend(ai_results.get('suggestions', []))

                # Update validation flags based on AI analysis
                if ai_results.get('syntax_issues'):
                    validation_results['syntax_valid'] = False
                if ai_results.get('structure_issues'):
                    validation_results['structure_valid'] = False
                if ai_results.get('semantic_issues'):
                    validation_results['semantic_valid'] = False

            except Exception as e:
                self.logger.warning(
                    "AI validation failed, using traditional validation only",
                    error=str(e)
                )

            # Traditional validation as backup/supplement
            # Syntax validation
            syntax_result = await self._validate_syntax(code, language)
            validation_results.update(syntax_result)

            # Structure validation
            if validation_results['syntax_valid']:
                structure_result = await self._validate_structure(code, language, strict_mode)
                validation_results.update(structure_result)

            # Semantic validation
            if validation_results['structure_valid']:
                semantic_result = await self._validate_semantics(code, language, strict_mode)
                validation_results.update(semantic_result)

            # Overall validation status
            validation_results['overall_valid'] = (
                validation_results['syntax_valid'] and
                validation_results['structure_valid'] and
                validation_results['semantic_valid']
            )

            # Remove duplicates
            validation_results['errors'] = list(set(validation_results['errors']))
            validation_results['warnings'] = list(set(validation_results['warnings']))
            validation_results['suggestions'] = list(set(validation_results['suggestions']))

            return validation_results

        except Exception as e:
            self.logger.error(f"Code validation failed: {e}", exc_info=True)
            return {
                'syntax_valid': False,
                'structure_valid': False,
                'semantic_valid': False,
                'overall_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'suggestions': []
            }

    # Private helper methods

    async def _analyze_generation_request(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """Analyze the code generation request to extract context and requirements."""
        context = {
            'language_features': self._get_language_features(request.language),
            'best_practices': self._get_language_best_practices(request.language),
            'common_patterns': self._get_common_patterns(request.language),
            'extracted_requirements': self._extract_requirements_from_description(request.description)
        }
        return context

    async def _generate_code_content(
        self,
        request: CodeGenerationRequest,
        analysis_context: Dict[str, Any]
    ) -> str:
        """Generate the actual code content using AI-powered generation."""
        try:
            # Determine complexity level based on request
            complexity = self._determine_complexity(request, analysis_context)

            # Create enhanced prompt with context
            enhanced_description = self._create_enhanced_prompt(request, analysis_context)

            # Use ModelManager for AI-powered code generation
            generated_code = await generate_code(
                description=enhanced_description,
                language=request.language.value.lower(),
                complexity=complexity
            )

            # Clean and format the generated code
            return self._clean_generated_code(generated_code, request.language)

        except Exception as e:
            self.logger.warning(
                "AI code generation failed, falling back to template-based generation",
                error=str(e),
                language=request.language.value
            )
            # Fallback to template-based generation
            if request.language == CodeLanguage.PYTHON:
                return await self._generate_python_code(request, analysis_context)
            elif request.language == CodeLanguage.JAVASCRIPT:
                return await self._generate_javascript_code(request, analysis_context)
            else:
                return await self._generate_generic_code(request, analysis_context)

    async def _generate_python_code(
        self,
        request: CodeGenerationRequest,
        analysis_context: Dict[str, Any]
    ) -> str:
        """Generate Python code based on the request."""
        # Extract function/class name from description
        description_lower = request.description.lower()

        if 'function' in description_lower or 'def' in description_lower:
            # Generate a function
            function_name = self._extract_function_name(request.description)
            parameters = self._extract_parameters(request.description)

            template = self.code_templates[CodeLanguage.PYTHON]['function']
            return template.format(
                function_name=function_name or 'generated_function',
                parameters=', '.join(parameters) if parameters else '',
                description=request.description,
                args_docs='\n        '.join([f"{param}: Description for {param}" for param in parameters]),
                return_docs="Description of return value",
                body="    # TODO: Implement function logic\n    pass"
            )

        elif 'class' in description_lower:
            # Generate a class
            class_name = self._extract_class_name(request.description)

            template = self.code_templates[CodeLanguage.PYTHON]['class']
            return template.format(
                class_name=class_name or 'GeneratedClass',
                description=request.description,
                init_params='',
                init_body='pass',
                methods='    def example_method(self):\n        """Example method."""\n        pass'
            )

        else:
            # Generate generic Python code
            return f'# {request.description}\n# TODO: Implement the requested functionality\npass'

    async def _generate_javascript_code(
        self,
        request: CodeGenerationRequest,
        analysis_context: Dict[str, Any]
    ) -> str:
        """Generate JavaScript code based on the request."""
        description_lower = request.description.lower()

        if 'function' in description_lower:
            function_name = self._extract_function_name(request.description)
            parameters = self._extract_parameters(request.description)
            param_list = ', '.join(parameters) if parameters else 'void'
            param_args = ', '.join(parameters) if parameters else ''

            return f'''/**
 * {request.description}
 * @param {{{param_list}}}
 * @returns {{*}} Description of return value
 */
function {function_name or 'generatedFunction'}({param_args}) {{
    // TODO: Implement function logic
}}'''

        elif 'class' in description_lower:
            class_name = self._extract_class_name(request.description)

            return f'''/**
 * {request.description}
 */
class {class_name or 'GeneratedClass'} {{
    constructor() {{
        // TODO: Initialize class properties
    }}

    exampleMethod() {{
        // TODO: Implement method
    }}
}}'''

        else:
            return f'// {request.description}\n// TODO: Implement the requested functionality'

    async def _generate_generic_code(
        self,
        request: CodeGenerationRequest,
        analysis_context: Dict[str, Any]
    ) -> str:
        """Generate generic code for other languages."""
        comment_prefix = {
            CodeLanguage.JAVA: '//',
            CodeLanguage.CPP: '//',
            CodeLanguage.C: '//',
            CodeLanguage.CSHARP: '//',
            CodeLanguage.GO: '//',
            CodeLanguage.RUST: '//',
            CodeLanguage.PHP: '//',
            CodeLanguage.RUBY: '#',
            CodeLanguage.SHELL: '#',
            CodeLanguage.SQL: '--',
        }.get(request.language, '#')

        return f'{comment_prefix} {request.description}\n{comment_prefix} TODO: Implement the requested functionality'

    async def _analyze_code(self, code: str, language: CodeLanguage) -> CodeAnalysis:
        """Analyze code quality and metrics using AI-powered analysis."""
        try:
            # Use AI-powered code analysis
            ai_analysis = await analyze_code(code, "comprehensive")

            # Parse AI analysis results
            analysis_results = self._parse_ai_analysis(ai_analysis, code, language)

            # Calculate basic metrics
            lines_of_code = len([line for line in code.split('\n') if line.strip()])

            # Combine AI analysis with basic metrics
            metrics = {
                'function_count': self._count_functions(code, language),
                'class_count': self._count_classes(code, language),
                'comment_ratio': self._calculate_comment_ratio(code, language),
                'avg_line_length': self._calculate_avg_line_length(code)
            }

            validation_passed = await self._quick_validation(code, language)

            # Use AI-derived values with fallbacks
            complexity_score = analysis_results.get('complexity_score', 5)
            quality_score = analysis_results.get('quality_score', 70)
            issues = analysis_results.get('issues', [])
            suggestions = analysis_results.get('suggestions', [])

        except Exception as e:
            self.logger.warning(
                "AI code analysis failed, falling back to basic analysis",
                error=str(e),
                language=language.value
            )
            # Fallback to basic analysis
            lines_of_code = len([line for line in code.split('\n') if line.strip()])
            complexity_score = await self._calculate_complexity(code, language)
            quality_score = await self._calculate_quality_score(code, language)
            issues = await self._identify_code_issues(code, language)
            suggestions = await self._generate_code_suggestions(code, language)

            metrics = {
                'function_count': self._count_functions(code, language),
                'class_count': self._count_classes(code, language),
                'comment_ratio': self._calculate_comment_ratio(code, language),
                'avg_line_length': self._calculate_avg_line_length(code)
            }

            validation_passed = await self._quick_validation(code, language)

            return CodeAnalysis(
                language=language,
                lines_of_code=lines_of_code,
                complexity_score=complexity_score,
                quality_score=quality_score,
                issues=issues,
                suggestions=suggestions,
                metrics=metrics,
                validation_passed=validation_passed
            )

        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return CodeAnalysis(
                language=language,
                lines_of_code=0,
                complexity_score=0.0,
                quality_score=0.0,
                issues=[{'type': 'analysis_error', 'message': str(e)}],
                suggestions=[],
                metrics={},
                validation_passed=False
            )

    # Message handlers

    async def _handle_generate_code(self, message: AgentMessage) -> None:
        """Handle code generation requests."""
        try:
            data = message.data
            request = CodeGenerationRequest(
                description=data.get('description', ''),
                language=CodeLanguage(data.get('language', 'python')),
                context=data.get('context'),
                requirements=data.get('requirements'),
                style_guide=data.get('style_guide'),
                existing_code=data.get('existing_code'),
                target_file=data.get('target_file')
            )

            result = await self.generate_code(request)

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='code_generation_result',
                data={
                    'generated_code': result.generated_code,
                    'analysis': result.analysis.__dict__,
                    'explanation': result.explanation,
                    'test_code': result.test_code,
                    'documentation': result.documentation,
                    'confidence_score': result.confidence_score
                },
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling generate_code: {e}", exc_info=True)

    async def _handle_modify_code(self, message: AgentMessage) -> None:
        """Handle code modification requests."""
        try:
            data = message.data
            result = await self.modify_code(
                existing_code=data.get('existing_code', ''),
                modifications=data.get('modifications', ''),
                language=CodeLanguage(data.get('language', 'python')),
                context=data.get('context')
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='code_modification_result',
                data=result.__dict__,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling modify_code: {e}", exc_info=True)

    async def _handle_review_code(self, message: AgentMessage) -> None:
        """Handle code review requests."""
        try:
            data = message.data
            result = await self.review_code(
                code=data.get('code', ''),
                language=CodeLanguage(data.get('language', 'python')),
                focus_areas=data.get('focus_areas')
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='code_review_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling review_code: {e}", exc_info=True)

    async def _handle_debug_code(self, message: AgentMessage) -> None:
        """Handle code debugging requests."""
        try:
            data = message.data
            result = await self.debug_code(
                code=data.get('code', ''),
                error_message=data.get('error_message'),
                language=CodeLanguage(data.get('language', 'python')),
                context=data.get('context')
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='code_debug_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling debug_code: {e}", exc_info=True)

    async def _handle_validate_code(self, message: AgentMessage) -> None:
        """Handle code validation requests."""
        try:
            data = message.data
            result = await self.validate_code(
                code=data.get('code', ''),
                language=CodeLanguage(data.get('language', 'python')),
                strict_mode=data.get('strict_mode', False)
            )

            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='code_validation_result',
                data=result,
                correlation_id=message.correlation_id
            )

            await self.send_message(response)

        except Exception as e:
            self.logger.error(f"Error handling validate_code: {e}", exc_info=True)

    async def _handle_task_request(self, message: AgentMessage) -> None:
        """Handle generic task requests from the orchestrator."""
        try:
            data = message.data
            action = data.get('action', '')

            if action == 'generate_code':
                await self._handle_generate_code(message)
            elif action == 'modify_code':
                await self._handle_modify_code(message)
            elif action == 'review_code':
                await self._handle_review_code(message)
            elif action == 'debug_code':
                await self._handle_debug_code(message)
            elif action == 'validate_code':
                await self._handle_validate_code(message)
            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            self.logger.error(f"Error handling task_request: {e}", exc_info=True)

    # Utility methods (simplified implementations)

    def _determine_complexity(self, request: CodeGenerationRequest, analysis_context: Dict[str, Any]) -> ComplexityLevel:
        """Determine the complexity level of the code generation request."""
        description_lower = request.description.lower()

        # Check for complexity indicators
        complex_keywords = ['algorithm', 'optimization', 'concurrent', 'async', 'distributed', 'machine learning', 'ai']
        medium_keywords = ['class', 'inheritance', 'database', 'api', 'framework', 'library']

        if any(keyword in description_lower for keyword in complex_keywords):
            return ComplexityLevel.COMPLEX
        elif any(keyword in description_lower for keyword in medium_keywords):
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE

    def _create_enhanced_prompt(self, request: CodeGenerationRequest, analysis_context: Dict[str, Any]) -> str:
        """Create an enhanced prompt with context for AI code generation."""
        base_prompt = request.description

        # Add language-specific requirements
        language_requirements = {
            CodeLanguage.PYTHON: "Follow PEP 8 style guide, use type hints, include docstrings",
            CodeLanguage.JAVASCRIPT: "Use modern ES6+ syntax, include JSDoc comments",
            CodeLanguage.TYPESCRIPT: "Use proper TypeScript types, include interface definitions",
            CodeLanguage.JAVA: "Follow Java naming conventions, include Javadoc comments",
            CodeLanguage.CPP: "Use modern C++ standards, include proper headers",
        }

        requirements = language_requirements.get(request.language, "Follow language best practices")

        enhanced_prompt = f"""{base_prompt}

Additional Requirements:
- {requirements}
- Include comprehensive error handling
- Add appropriate logging where needed
- Write clean, maintainable code
- Include unit tests if applicable
- Optimize for readability and performance

Context: {analysis_context.get('context', 'Standard development environment')}"""

        return enhanced_prompt

    def _clean_generated_code(self, generated_code: str, language: CodeLanguage) -> str:
        """Clean and format the generated code."""
        # Remove common AI response artifacts
        code = generated_code.strip()

        # Remove markdown code blocks if present
        if code.startswith('```'):
            lines = code.split('\n')
            # Remove first line (```language) and last line (```)
            if lines[-1].strip() == '```':
                lines = lines[1:-1]
            elif len(lines) > 1:
                lines = lines[1:]  # Just remove first line
            code = '\n'.join(lines)

        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the code:",
            "Here is the code:",
            "The code is:",
            "Code:",
            "Solution:",
        ]

        for prefix in prefixes_to_remove:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()

        # Language-specific cleaning
        if language == CodeLanguage.PYTHON:
            # Ensure proper indentation
            code = self._fix_python_indentation(code)

        return code.strip()

    def _fix_python_indentation(self, code: str) -> str:
        """Fix Python indentation issues."""
        try:
            # Try to parse and reformat using ast
            tree = ast.parse(code)
            # If it parses successfully, return as-is
            return code
        except SyntaxError:
            # If there are syntax errors, try basic indentation fixing
            lines = code.split('\n')
            fixed_lines = []
            current_indent = 0

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    fixed_lines.append('')
                    continue

                # Adjust indentation based on keywords
                if any(stripped.startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:']):
                    fixed_lines.append('    ' * current_indent + stripped)
                    if stripped.endswith(':'):
                        current_indent += 1
                elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                    if current_indent > 0:
                        current_indent -= 1
                    fixed_lines.append('    ' * current_indent + stripped)
                    current_indent += 1
                elif stripped == 'pass':
                    fixed_lines.append('    ' * current_indent + stripped)
                    if current_indent > 0:
                        current_indent -= 1
                else:
                    fixed_lines.append('    ' * current_indent + stripped)

            return '\n'.join(fixed_lines)

    def _parse_ai_analysis(self, ai_analysis: str, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Parse AI analysis results and extract structured information."""
        results = {
            'complexity_score': 5,
            'quality_score': 70,
            'issues': [],
            'suggestions': []
        }

        try:
            analysis_lower = ai_analysis.lower()

            # Extract complexity score (look for numbers 1-10)
            complexity_patterns = [
                r'complexity[:\s]*(\d+)',
                r'complexity score[:\s]*(\d+)',
                r'cyclomatic complexity[:\s]*(\d+)'
            ]

            for pattern in complexity_patterns:
                match = re.search(pattern, analysis_lower)
                if match:
                    score = int(match.group(1))
                    results['complexity_score'] = min(max(score, 1), 10)
                    break

            # Extract quality score (look for percentages or scores out of 100)
            quality_patterns = [
                r'quality[:\s]*(\d+)%',
                r'quality score[:\s]*(\d+)',
                r'code quality[:\s]*(\d+)'
            ]

            for pattern in quality_patterns:
                match = re.search(pattern, analysis_lower)
                if match:
                    score = int(match.group(1))
                    results['quality_score'] = min(max(score, 0), 100)
                    break

            # Extract issues (look for bullet points, numbered lists, or "issue:" patterns)
            issue_patterns = [
                r'(?:issues?|problems?|bugs?)[:\n]([^\n]+)',
                r'[-*•]\s*([^\n]+(?:issue|problem|bug|error))',
                r'\d+\.\s*([^\n]+(?:issue|problem|bug|error))'
            ]

            for pattern in issue_patterns:
                matches = re.findall(pattern, ai_analysis, re.IGNORECASE)
                results['issues'].extend([match.strip() for match in matches])

            # Extract suggestions (look for recommendations, improvements)
            suggestion_patterns = [
                r'(?:suggestions?|recommendations?|improvements?)[:\n]([^\n]+)',
                r'[-*•]\s*([^\n]+(?:suggest|recommend|improve|consider))',
                r'\d+\.\s*([^\n]+(?:suggest|recommend|improve|consider))'
            ]

            for pattern in suggestion_patterns:
                matches = re.findall(pattern, ai_analysis, re.IGNORECASE)
                results['suggestions'].extend([match.strip() for match in matches])

            # Remove duplicates and empty entries
            results['issues'] = list(set(filter(None, results['issues'])))
            results['suggestions'] = list(set(filter(None, results['suggestions'])))

        except Exception as e:
            self.logger.warning(f"Error parsing AI analysis: {e}")

        return results

    def _parse_ai_validation(self, ai_validation: str, language: CodeLanguage) -> Dict[str, Any]:
        """Parse AI validation results and extract structured validation information."""
        results = {
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'syntax_issues': False,
            'structure_issues': False,
            'semantic_issues': False
        }

        try:
            validation_lower = ai_validation.lower()

            # Check for syntax issues
            syntax_indicators = ['syntax error', 'parse error', 'invalid syntax', 'compilation error']
            if any(indicator in validation_lower for indicator in syntax_indicators):
                results['syntax_issues'] = True

            # Check for structure issues
            structure_indicators = ['structure', 'organization', 'architecture', 'design pattern']
            if any(indicator in validation_lower for indicator in structure_indicators):
                results['structure_issues'] = True

            # Check for semantic issues
            semantic_indicators = ['logic error', 'semantic', 'behavior', 'runtime error']
            if any(indicator in validation_lower for indicator in semantic_indicators):
                results['semantic_issues'] = True

            # Extract errors
            error_patterns = [
                r'error[:\s]*([^\n]+)',
                r'[-*•]\s*([^\n]+(?:error|fail|wrong|incorrect))',
                r'(?:critical|fatal)[:\s]*([^\n]+)'
            ]

            for pattern in error_patterns:
                matches = re.findall(pattern, ai_validation, re.IGNORECASE)
                results['errors'].extend([match.strip() for match in matches])

            # Extract warnings
            warning_patterns = [
                r'warning[:\s]*([^\n]+)',
                r'[-*•]\s*([^\n]+(?:warning|caution|note))',
                r'(?:potential|possible)[:\s]*([^\n]+)'
            ]

            for pattern in warning_patterns:
                matches = re.findall(pattern, ai_validation, re.IGNORECASE)
                results['warnings'].extend([match.strip() for match in matches])

            # Extract suggestions
            suggestion_patterns = [
                r'(?:suggestion|recommendation)[:\s]*([^\n]+)',
                r'[-*•]\s*([^\n]+(?:should|could|consider|try))',
                r'(?:improve|optimize)[:\s]*([^\n]+)'
            ]

            for pattern in suggestion_patterns:
                matches = re.findall(pattern, ai_validation, re.IGNORECASE)
                results['suggestions'].extend([match.strip() for match in matches])

            # Clean up results
            for key in ['errors', 'warnings', 'suggestions']:
                results[key] = [item for item in results[key] if item and len(item) > 10]

        except Exception as e:
            self.logger.warning(f"Error parsing AI validation: {e}")

        return results

    def _extract_function_name(self, description: str) -> Optional[str]:
        """Extract function name from description."""
        patterns = [
            r'function\s+(\w+)',
            r'def\s+(\w+)',
            r'create\s+(\w+)',
            r'(\w+)\s+function'
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)

    # Abstract method implementations
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and route it to appropriate code handlers.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        try:
            self.logger.debug(f"Processing message: {message.message_type} from {message.sender_id}")

            # Handle different message types
            if message.message_type == "code_request":
                payload = message.payload
                action = payload.get('action', 'generate')

                if action == "generate":
                    result = await self.generate_code(
                        description=payload.get('description', ''),
                        language=payload.get('language', 'python'),
                        requirements=payload.get('requirements', [])
                    )
                elif action == "modify":
                    result = await self.modify_code(
                        original_code=payload.get('original_code', ''),
                        modifications=payload.get('modifications', ''),
                        language=payload.get('language', 'python')
                    )
                elif action == "review":
                    result = await self.review_code(
                        code=payload.get('code', ''),
                        language=payload.get('language', 'python'),
                        focus_areas=payload.get('focus_areas', [])
                    )
                elif action == "debug":
                    result = await self.debug_code(
                        code=payload.get('code', ''),
                        error_message=payload.get('error_message', ''),
                        language=payload.get('language', 'python')
                    )
                else:
                    raise ValueError(f"Unknown code action: {action}")

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="code_response",
                    payload=result,
                    correlation_id=message.correlation_id
                )

            elif message.message_type == "task_request":
                await self._handle_task_request(message)
                return None

            else:
                # Return None for unhandled message types to avoid recursion
                self.logger.debug(f"Unhandled message type: {message.message_type}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)

            # Return error response if this was a request expecting a response
            if message.message_type in ["code_request"]:
                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="error_response",
                    payload={
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'failed_action': message.payload.get('action', 'unknown')
                    },
                    correlation_id=message.correlation_id
                )

            return None

    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return list of capabilities this code agent provides.

        Returns:
            List of agent capabilities
        """


        return [
            AgentCapability(
                name="code_generation",
                description="Generate code from natural language descriptions",
                input_types=["text"],
                output_types=["code"],
                cost_estimate=2.0
            ),
            AgentCapability(
                name="code_modification",
                description="Modify and refactor existing code",
                input_types=["code", "text"],
                output_types=["code"],
                cost_estimate=1.5
            ),
            AgentCapability(
                name="code_review",
                description="Review code for quality, style, and potential issues",
                input_types=["code"],
                output_types=["text", "dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="code_debugging",
                description="Debug code and suggest fixes for errors",
                input_types=["code", "text"],
                output_types=["code", "text"],
                cost_estimate=2.5
            ),
            AgentCapability(
                name="syntax_validation",
                description="Validate code syntax for multiple programming languages",
                input_types=["code"],
                output_types=["dict"],
                cost_estimate=0.5
            ),
            AgentCapability(
                name="code_analysis",
                description="Analyze code complexity, performance, and structure",
                input_types=["code"],
                output_types=["dict"],
                cost_estimate=1.5
            ),
            AgentCapability(
                name="multi_language_support",
                description="Support for Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more",
                input_types=["code"],
                output_types=["code"],
                cost_estimate=1.0
            )
        ]

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status information
        """
        try:
            # Initialize base health status
            base_health = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "state": self.state.value if hasattr(self, 'state') else "unknown"
            }

            # Check code agent-specific health
            current_time = datetime.utcnow()

            # Check supported languages
            languages_available = list(getattr(self, 'language_templates', {}).keys())
            language_health = len(languages_available) > 0

            # Check code generation capabilities
            can_generate = hasattr(self, 'generate_code')
            can_modify = hasattr(self, 'modify_code')
            can_review = hasattr(self, 'review_code')
            can_debug = hasattr(self, 'debug_code')

            core_capabilities = can_generate and can_modify and can_review and can_debug

            # Calculate performance metrics
            performance_metrics = getattr(self, 'performance_metrics', {})
            total_tasks = performance_metrics.get('tasks_completed', 0) + performance_metrics.get('tasks_failed', 0)
            success_rate = 0.0
            if total_tasks > 0:
                success_rate = (performance_metrics.get('tasks_completed', 0) / total_tasks) * 100

            # Determine overall health status
            is_healthy = (
                language_health and
                core_capabilities and
                success_rate >= 80.0  # Consider healthy if success rate >= 80%
            )

            health_status = {
                **base_health,
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": current_time.isoformat(),
                "code_agent_specific": {
                    "supported_languages": languages_available,
                    "language_count": len(languages_available),
                    "language_health": language_health,
                    "core_capabilities": {
                        "generate": can_generate,
                        "modify": can_modify,
                        "review": can_review,
                        "debug": can_debug
                    },
                    "capabilities_health": core_capabilities,
                    "success_rate": success_rate,
                    "tasks_completed": performance_metrics.get('tasks_completed', 0),
                    "tasks_failed": performance_metrics.get('tasks_failed', 0),
                    "average_response_time": performance_metrics.get('avg_response_time', 0.0)
                },
                "capabilities": len(self.get_capabilities()),
                "uptime": (current_time - self.created_at).total_seconds() if hasattr(self, 'created_at') else 0
            }

            # Add any critical issues
            issues = []
            if not language_health:
                issues.append("No programming languages available")
            if not core_capabilities:
                issues.append("Missing core code manipulation capabilities")
            if success_rate < 80.0 and total_tasks > 0:
                issues.append(f"Low success rate: {success_rate:.1f}%")

            if issues:
                health_status["issues"] = issues

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }

    # Missing helper methods for AI-enhanced code generation

    def _extract_parameters(self, description: str) -> List[str]:
        """Extract function parameters from description."""
        patterns = [
            r'parameters?\s*:\s*([^.]+)',
            r'takes?\s+([^.]+)\s+as\s+(?:parameters?|arguments?)',
            r'with\s+(?:parameters?|arguments?)\s+([^.]+)',
            r'\(([^)]+)\)'
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                params_str = match.group(1)
                # Clean and split parameters
                params = [p.strip() for p in re.split(r'[,\s]+', params_str) if p.strip()]
                return params[:5]  # Limit to 5 parameters

        return []

    def _extract_class_name(self, description: str) -> Optional[str]:
        """Extract class name from description."""
        patterns = [
            r'class\s+(\w+)',
            r'create\s+(?:a\s+)?(\w+)\s+class',
            r'(\w+)\s+class',
            r'implement\s+(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                class_name = match.group(1)
                # Capitalize first letter
                return class_name[0].upper() + class_name[1:] if class_name else None

        return None

    async def _generate_test_code(self, generated_code: str, request: CodeGenerationRequest) -> Optional[str]:
        """Generate test code for the generated code."""
        try:
            if request.language == CodeLanguage.PYTHON:
                # Extract function/class name from generated code
                function_match = re.search(r'def\s+(\w+)', generated_code)
                class_match = re.search(r'class\s+(\w+)', generated_code)

                if function_match:
                    func_name = function_match.group(1)
                    return f'''import unittest
from unittest.mock import Mock, patch

class Test{func_name.title()}(unittest.TestCase):
    """Test cases for {func_name} function."""

    def test_{func_name}_basic(self):
        """Test basic functionality of {func_name}."""
        # TODO: Add test implementation
        pass

    def test_{func_name}_edge_cases(self):
        """Test edge cases for {func_name}."""
        # TODO: Add edge case tests
        pass

if __name__ == '__main__':
    unittest.main()'''

                elif class_match:
                    class_name = class_match.group(1)
                    return f'''import unittest
from unittest.mock import Mock, patch

class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name} class."""

    def setUp(self):
        """Set up test fixtures."""
        self.instance = {class_name}()

    def test_initialization(self):
        """Test class initialization."""
        # TODO: Add initialization tests
        pass

if __name__ == '__main__':
    unittest.main()'''

            elif request.language == CodeLanguage.JAVASCRIPT:
                function_match = re.search(r'function\s+(\w+)', generated_code)
                if function_match:
                    func_name = function_match.group(1)
                    return f'''const {{ {func_name} }} = require('./your-module');

describe('{func_name}', () => {{
    test('should work with valid input', () => {{
        // TODO: Add test implementation
    }});

    test('should handle edge cases', () => {{
        // TODO: Add edge case tests
    }});
}});'''

            return None

        except Exception as e:
            self.logger.warning(f"Failed to generate test code: {e}")
            return None

    async def _generate_documentation(self, generated_code: str, request: CodeGenerationRequest) -> str:
        """Generate documentation for the generated code."""
        try:
            doc_sections = [
                f"# {request.description}",
                "",
                "## Overview",
                f"This code implements: {request.description}",
                "",
                "## Usage",
                "```" + request.language.value.lower(),
                generated_code.strip(),
                "```",
                "",
                "## Features",
                "- Clean, readable code structure",
                "- Proper error handling",
                "- Follow language best practices",
                "",
                "## Requirements",
                f"- {request.language.value} runtime environment"
            ]

            return "\n".join(doc_sections)

        except Exception as e:
            self.logger.warning(f"Failed to generate documentation: {e}")
            return f"Documentation for: {request.description}"

    def _calculate_confidence_score(self, code_analysis: CodeAnalysis, request: CodeGenerationRequest) -> float:
        """Calculate confidence score based on code analysis and request complexity."""
        try:
            base_score = 70.0  # Base confidence

            # Adjust based on code quality
            quality_bonus = (code_analysis.quality_score - 50) * 0.4  # +/- 20 points max

            # Adjust based on complexity
            if code_analysis.complexity_score <= 3:
                complexity_bonus = 10.0  # Simple code, high confidence
            elif code_analysis.complexity_score <= 7:
                complexity_bonus = 0.0   # Medium complexity, neutral
            else:
                complexity_bonus = -10.0  # High complexity, lower confidence

            # Adjust based on validation
            validation_bonus = 15.0 if code_analysis.validation_passed else -15.0

            # Adjust based on language familiarity
            language_bonus = 5.0 if request.language in [CodeLanguage.PYTHON, CodeLanguage.JAVASCRIPT] else 0.0

            # Calculate final score
            confidence = base_score + quality_bonus + complexity_bonus + validation_bonus + language_bonus

            # Clamp to 0-100 range
            return max(0.0, min(100.0, confidence))

        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence score: {e}")
            return 50.0  # Default moderate confidence

    def _generate_explanation(self, generated_code: str, request: CodeGenerationRequest, analysis_context: Dict[str, Any]) -> str:
        """Generate explanation of the generated code."""
        try:
            lines = generated_code.split('\n')
            code_length = len([line for line in lines if line.strip()])

            explanation_parts = [
                f"Generated {request.language.value} code for: {request.description}",
                "",
                f"Code Statistics:",
                f"- Lines of code: {code_length}",
                f"- Language: {request.language.value}",
                f"- Complexity: {analysis_context.get('complexity', 'Medium')}",
                "",
                "Key Features:",
            ]

            # Add language-specific features
            if request.language == CodeLanguage.PYTHON:
                if 'def ' in generated_code:
                    explanation_parts.append("- Function-based implementation")
                if 'class ' in generated_code:
                    explanation_parts.append("- Object-oriented design")
                if '"""' in generated_code:
                    explanation_parts.append("- Comprehensive docstrings")
                if 'try:' in generated_code:
                    explanation_parts.append("- Error handling included")

            elif request.language == CodeLanguage.JAVASCRIPT:
                if 'async ' in generated_code:
                    explanation_parts.append("- Asynchronous operations support")
                if 'const ' in generated_code:
                    explanation_parts.append("- Modern ES6+ syntax")
                if 'try {' in generated_code:
                    explanation_parts.append("- Error handling included")

            explanation_parts.extend([
                "",
                "This code follows best practices and includes appropriate error handling where applicable."
            ])

            return "\n".join(explanation_parts)

        except Exception as e:
            self.logger.warning(f"Failed to generate explanation: {e}")
            return f"Generated code implementation for: {request.description}"

    def _update_generation_metrics(self, language: CodeLanguage, generation_time: float, success: bool):
        """Update generation performance metrics."""
        try:
            if not hasattr(self, 'generation_metrics'):
                self.generation_metrics = {
                    'total_requests': 0,
                    'successful_generations': 0,
                    'failed_generations': 0,
                    'average_generation_time': 0.0,
                    'language_distribution': {},
                    'quality_distribution': {}
                }

            # Update counters
            self.generation_metrics['total_requests'] += 1

            if success:
                self.generation_metrics['successful_generations'] += 1
            else:
                self.generation_metrics['failed_generations'] += 1

            # Update language distribution
            lang_key = language.value
            if lang_key not in self.generation_metrics['language_distribution']:
                self.generation_metrics['language_distribution'][lang_key] = 0
            self.generation_metrics['language_distribution'][lang_key] += 1

            # Update average generation time
            total_requests = self.generation_metrics['total_requests']
            current_avg = self.generation_metrics['average_generation_time']
            self.generation_metrics['average_generation_time'] = (
                (current_avg * (total_requests - 1) + generation_time) / total_requests
            )

        except Exception as e:
            self.logger.warning(f"Failed to update generation metrics: {e}")

    def _get_language_features(self, language: CodeLanguage) -> List[str]:
        """Get key features of the specified programming language."""
        features_map = {
            CodeLanguage.PYTHON: [
                'dynamic typing', 'indentation-based blocks', 'list comprehensions',
                'decorators', 'context managers', 'generators', 'duck typing'
            ],
            CodeLanguage.JAVASCRIPT: [
                'prototypal inheritance', 'closures', 'hoisting', 'promises',
                'arrow functions', 'destructuring', 'template literals'
            ],
            CodeLanguage.TYPESCRIPT: [
                'static typing', 'interfaces', 'generics', 'decorators',
                'modules', 'union types', 'type guards'
            ],
            CodeLanguage.JAVA: [
                'static typing', 'inheritance', 'interfaces', 'generics',
                'annotations', 'packages', 'exception handling'
            ],
            CodeLanguage.CPP: [
                'manual memory management', 'templates', 'operator overloading',
                'multiple inheritance', 'RAII', 'const correctness'
            ]
        }
        return features_map.get(language, ['general programming constructs'])

    def _get_language_best_practices(self, language: CodeLanguage) -> List[str]:
        """Get best practices for the specified programming language."""
        practices_map = {
            CodeLanguage.PYTHON: [
                'Follow PEP 8 style guide', 'Use descriptive variable names',
                'Write docstrings for functions and classes', 'Use list comprehensions appropriately',
                'Handle exceptions properly', 'Use type hints'
            ],
            CodeLanguage.JAVASCRIPT: [
                'Use const/let instead of var', 'Use arrow functions appropriately',
                'Handle promises with async/await', 'Use strict mode',
                'Avoid global variables', 'Use JSDoc for documentation'
            ],
            CodeLanguage.TYPESCRIPT: [
                'Define proper interfaces', 'Use strict type checking',
                'Avoid any type when possible', 'Use generics for reusability',
                'Organize code with modules', 'Use readonly for immutable data'
            ],
            CodeLanguage.JAVA: [
                'Follow naming conventions', 'Use proper access modifiers',
                'Handle checked exceptions', 'Use generics for type safety',
                'Follow SOLID principles', 'Write Javadoc comments'
            ],
            CodeLanguage.CPP: [
                'Use RAII for resource management', 'Prefer const correctness',
                'Use smart pointers', 'Follow rule of three/five',
                'Use namespaces appropriately', 'Avoid memory leaks'
            ]
        }
        return practices_map.get(language, ['Write clean, readable code'])

    def _get_common_patterns(self, language: CodeLanguage) -> List[str]:
        """Get common design patterns for the specified programming language."""
        patterns_map = {
            CodeLanguage.PYTHON: [
                'Factory pattern', 'Singleton pattern', 'Decorator pattern',
                'Observer pattern', 'Context manager pattern', 'Iterator pattern'
            ],
            CodeLanguage.JAVASCRIPT: [
                'Module pattern', 'Observer pattern', 'Factory pattern',
                'Singleton pattern', 'Promise pattern', 'Callback pattern'
            ],
            CodeLanguage.TYPESCRIPT: [
                'Factory pattern', 'Builder pattern', 'Strategy pattern',
                'Decorator pattern', 'Dependency injection pattern'
            ],
            CodeLanguage.JAVA: [
                'Factory pattern', 'Builder pattern', 'Strategy pattern',
                'Observer pattern', 'MVC pattern', 'Dependency injection'
            ],
            CodeLanguage.CPP: [
                'RAII pattern', 'Factory pattern', 'Strategy pattern',
                'Template method pattern', 'Singleton pattern'
            ]
        }
        return patterns_map.get(language, ['Basic design patterns'])

    def _extract_requirements_from_description(self, description: str) -> List[str]:
        """Extract specific requirements from the description."""
        requirements = []
        desc_lower = description.lower()

        # Common requirement indicators
        if 'error handling' in desc_lower or 'handle errors' in desc_lower:
            requirements.append('error_handling')
        if 'async' in desc_lower or 'asynchronous' in desc_lower:
            requirements.append('async_support')
        if 'test' in desc_lower:
            requirements.append('testable')
        if 'validate' in desc_lower or 'validation' in desc_lower:
            requirements.append('input_validation')
        if 'log' in desc_lower or 'logging' in desc_lower:
            requirements.append('logging')
        if 'config' in desc_lower or 'configuration' in desc_lower:
            requirements.append('configurable')
        if 'thread' in desc_lower or 'concurrent' in desc_lower:
            requirements.append('thread_safe')
        if 'database' in desc_lower or 'db' in desc_lower:
            requirements.append('database_integration')
        if 'api' in desc_lower or 'rest' in desc_lower:
            requirements.append('api_integration')
        if 'security' in desc_lower or 'secure' in desc_lower:
            requirements.append('security_focused')

        return requirements

    async def _validate_syntax(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Validate code syntax."""
        result = {
            'syntax_valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            if language == CodeLanguage.PYTHON:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    result['syntax_valid'] = False
                    result['errors'].append(f"Python syntax error: {str(e)}")
            elif language == CodeLanguage.JAVASCRIPT:
                # Basic JavaScript validation (simplified)
                if 'function(' in code and not code.count('(') == code.count(')'):
                    result['syntax_valid'] = False
                    result['errors'].append("Mismatched parentheses")
                if '{' in code and not code.count('{') == code.count('}'):
                    result['syntax_valid'] = False
                    result['errors'].append("Mismatched braces")
            # Add more language validations as needed

        except Exception as e:
            result['syntax_valid'] = False
            result['errors'].append(f"Syntax validation error: {str(e)}")

        return result

    async def _validate_structure(self, code: str, language: CodeLanguage, strict_mode: bool) -> Dict[str, Any]:
        """Validate code structure."""
        result = {
            'structure_valid': True,
            'warnings': [],
            'suggestions': []
        }

        try:
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            # Check for minimum structure
            if len(non_empty_lines) == 0:
                result['structure_valid'] = False
                result['warnings'].append("Empty code structure")
                return result

            # Language-specific structure validation
            if language == CodeLanguage.PYTHON:
                # Check for proper indentation patterns
                indent_levels = []
                for line in lines:
                    if line.strip():
                        leading_spaces = len(line) - len(line.lstrip())
                        indent_levels.append(leading_spaces)

                # Check for consistent indentation
                if len(set(indent_levels)) > 1:
                    indent_steps = set()
                    for i in range(1, len(indent_levels)):
                        if indent_levels[i] != indent_levels[i-1]:
                            indent_steps.add(abs(indent_levels[i] - indent_levels[i-1]))

                    if len(indent_steps) > 1:
                        result['warnings'].append("Inconsistent indentation detected")

            elif language == CodeLanguage.JAVASCRIPT:
                # Check for proper brace usage
                if 'function' in code and '{' not in code:
                    result['warnings'].append("Function declaration without braces")

            if strict_mode:
                # Add stricter validation rules
                if language == CodeLanguage.PYTHON:
                    if 'def ' in code and '"""' not in code and "'''" not in code:
                        result['suggestions'].append("Consider adding docstrings to functions")

        except Exception as e:
            result['warnings'].append(f"Structure validation error: {str(e)}")

        return result

    async def _validate_semantics(self, code: str, language: CodeLanguage, strict_mode: bool) -> Dict[str, Any]:
        """Validate code semantics."""
        result = {
            'semantic_valid': True,
            'warnings': [],
            'suggestions': []
        }

        try:
            # Basic semantic checks
            if language == CodeLanguage.PYTHON:
                # Check for common semantic issues
                if 'return' in code and 'def ' not in code and 'class ' not in code:
                    result['warnings'].append("Return statement outside function")

                # Check for undefined variables (basic check)
                if 'print(' in code:
                    # This is a very basic check - in practice would need proper AST analysis
                    pass

            elif language == CodeLanguage.JAVASCRIPT:
                # Check for common JavaScript semantic issues
                if 'var ' in code:
                    result['suggestions'].append("Consider using 'let' or 'const' instead of 'var'")

            # Check for common anti-patterns
            if 'TODO' in code or 'FIXME' in code:
                result['warnings'].append("Code contains TODO/FIXME comments")

        except Exception as e:
            result['warnings'].append(f"Semantic validation error: {str(e)}")

        return result

    async def _quick_validation(self, code: str, language: CodeLanguage) -> bool:
        """Perform quick validation to check if code is syntactically correct."""
        try:
            if language == CodeLanguage.PYTHON:
                try:
                    ast.parse(code)
                    return True
                except SyntaxError:
                    return False
            elif language == CodeLanguage.JAVASCRIPT:
                # Basic JavaScript validation
                return (code.count('(') == code.count(')') and
                       code.count('{') == code.count('}') and
                       code.count('[') == code.count(']'))
            else:
                # For other languages, assume valid if not empty
                return bool(code.strip())

        except Exception:
            return False

    async def _calculate_complexity(self, code: str, language: CodeLanguage) -> int:
        """Calculate code complexity score (1-10)."""
        try:
            complexity = 1

            # Count control structures
            control_patterns = ['if ', 'for ', 'while ', 'switch ', 'case ', 'elif ', 'else:']
            for pattern in control_patterns:
                complexity += code.lower().count(pattern)

            # Count function definitions
            if language == CodeLanguage.PYTHON:
                complexity += code.count('def ')
            elif language == CodeLanguage.JAVASCRIPT:
                complexity += code.count('function ')

            # Count nested structures (simple approximation)
            lines = code.split('\n')
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent // 4)  # Assuming 4-space indents

            complexity += max_indent

            return min(complexity, 10)  # Cap at 10

        except Exception:
            return 5  # Default medium complexity

    async def _calculate_quality_score(self, code: str, language: CodeLanguage) -> int:
        """Calculate code quality score (0-100)."""
        try:
            score = 70  # Base score

            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            if not non_empty_lines:
                return 0

            # Check for comments/documentation
            comment_count = 0
            if language == CodeLanguage.PYTHON:
                comment_count = sum(1 for line in lines if '#' in line or '"""' in line or "'''" in line)
            elif language == CodeLanguage.JAVASCRIPT:
                comment_count = sum(1 for line in lines if '//' in line or '/*' in line)

            comment_ratio = comment_count / len(non_empty_lines)
            if comment_ratio > 0.2:
                score += 10
            elif comment_ratio > 0.1:
                score += 5

            # Check line length
            long_lines = [line for line in lines if len(line) > 120]
            if len(long_lines) / len(non_empty_lines) > 0.2:
                score -= 10

            # Check for good practices
            if language == CodeLanguage.PYTHON:
                if 'def ' in code and '"""' in code:
                    score += 10  # Has functions with docstrings
                if 'try:' in code and 'except' in code:
                    score += 5   # Has error handling

            elif language == CodeLanguage.JAVASCRIPT:
                if 'const ' in code or 'let ' in code:
                    score += 5   # Uses modern variable declarations
                if 'async ' in code and 'await ' in code:
                    score += 5   # Uses modern async patterns

            return max(0, min(100, score))

        except Exception:
            return 50  # Default moderate quality

    async def _identify_code_issues(self, code: str, language: CodeLanguage) -> List[str]:
        """Identify potential issues in the code."""
        issues = []

        try:
            lines = code.split('\n')

            # Check for common issues
            if any(len(line) > 120 for line in lines):
                issues.append("Lines exceed recommended length (120 chars)")

            if language == CodeLanguage.PYTHON:
                if 'print(' in code:
                    issues.append("Contains print statements (consider using logging)")
                if code.count('def ') > 10:
                    issues.append("High number of functions in single file")
                if any('import *' in line for line in lines):
                    issues.append("Uses wildcard imports")

            elif language == CodeLanguage.JAVASCRIPT:
                if 'var ' in code:
                    issues.append("Uses 'var' instead of 'let' or 'const'")
                if '==' in code and '===' not in code:
                    issues.append("Uses loose equality (==) instead of strict equality (===)")

            # Check for TODO/FIXME
            if any(keyword in code.upper() for keyword in ['TODO', 'FIXME', 'HACK']):
                issues.append("Contains TODO/FIXME comments")

        except Exception as e:
            issues.append(f"Issue analysis error: {str(e)}")

        return issues

    async def _generate_code_suggestions(self, code: str, language: CodeLanguage) -> List[str]:
        """Generate suggestions for code improvement."""
        suggestions = []

        try:
            if language == CodeLanguage.PYTHON:
                if 'def ' in code and '"""' not in code:
                    suggestions.append("Add docstrings to functions for better documentation")
                if 'print(' in code:
                    suggestions.append("Replace print statements with proper logging")
                if not any('try:' in line for line in code.split('\n')):
                    suggestions.append("Consider adding error handling with try/except blocks")

            elif language == CodeLanguage.JAVASCRIPT:
                if 'var ' in code:
                    suggestions.append("Replace 'var' with 'let' or 'const' for better scoping")
                if 'function(' in code and '=>' not in code:
                    suggestions.append("Consider using arrow functions for cleaner syntax")
                if not any('try {' in line for line in code.split('\n')):
                    suggestions.append("Consider adding error handling with try/catch blocks")

            # General suggestions
            lines = code.split('\n')
            if any(len(line) > 100 for line in lines):
                suggestions.append("Consider breaking long lines for better readability")

        except Exception as e:
            suggestions.append(f"Suggestion generation error: {str(e)}")

        return suggestions

    def _count_functions(self, code: str, language: CodeLanguage) -> int:
        """Count the number of functions in the code."""
        try:
            if language == CodeLanguage.PYTHON:
                return code.count('def ')
            elif language == CodeLanguage.JAVASCRIPT:
                return code.count('function ') + code.count(' => ')
            elif language == CodeLanguage.JAVA:
                # Simple approximation
                return len([line for line in code.split('\n') if 'public ' in line and '(' in line])
            else:
                return 0
        except Exception:
            return 0

    def _count_classes(self, code: str, language: CodeLanguage) -> int:
        """Count the number of classes in the code."""
        try:
            if language == CodeLanguage.PYTHON:
                return code.count('class ')
            elif language == CodeLanguage.JAVASCRIPT:
                return code.count('class ')
            elif language == CodeLanguage.JAVA:
                return code.count('class ') + code.count('interface ')
            else:
                return 0
        except Exception:
            return 0

    def _calculate_comment_ratio(self, code: str, language: CodeLanguage) -> float:
        """Calculate the ratio of comment lines to total lines."""
        try:
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            if not non_empty_lines:
                return 0.0

            comment_lines = 0
            if language == CodeLanguage.PYTHON:
                comment_lines = len([line for line in lines if line.strip().startswith('#') or '"""' in line or "'''" in line])
            elif language == CodeLanguage.JAVASCRIPT:
                comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
            elif language in [CodeLanguage.JAVA, CodeLanguage.CPP, CodeLanguage.C]:
                comment_lines = len([line for line in lines if line.strip().startswith('//') or '/*' in line or '*/' in line])
            else:
                # Generic comment detection
                comment_lines = len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')])

            return comment_lines / len(non_empty_lines)

        except Exception:
            return 0.0

    def _calculate_avg_line_length(self, code: str) -> float:
        """Calculate the average line length."""
        try:
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            if not non_empty_lines:
                return 0.0

            total_length = sum(len(line) for line in non_empty_lines)
            return total_length / len(non_empty_lines)

        except Exception:
            return 0.0
