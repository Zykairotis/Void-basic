"""
Strategy pattern implementation for different edit operations in Aider.

This module provides a flexible architecture for handling various edit formats
using the Strategy design pattern, enabling easy extensibility and maintainability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import re
import difflib

from .exceptions import (
    EditOperationError,
    SearchTextNotFoundError,
    SearchTextNotUniqueError,
    DiffApplicationError,
    MalformedEditError,
    ValidationError,
    handle_edit_errors
)


@dataclass
class EditResult:
    """Result of an edit operation."""
    success: bool
    file_path: str
    original_content: str
    new_content: str
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EditInstruction:
    """Represents an edit instruction with all necessary context."""
    file_path: str
    edit_type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# Strategy Interface
# =============================================================================

class EditStrategy(ABC):
    """
    Abstract base class for all edit strategies.

    This defines the interface that all concrete edit strategies must implement,
    enabling the Strategy pattern for handling different edit formats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @property
    @abstractmethod
    def edit_format(self) -> str:
        """Return the edit format identifier."""
        pass

    @property
    @abstractmethod
    def supported_operations(self) -> List[str]:
        """Return list of supported operations (create, update, delete)."""
        pass

    @abstractmethod
    def validate_edit_instruction(self, instruction: EditInstruction) -> bool:
        """
        Validate an edit instruction before processing.

        Args:
            instruction: The edit instruction to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        pass

    @abstractmethod
    def parse_edits(self, content: str) -> List[EditInstruction]:
        """
        Parse edit instructions from AI response content.

        Args:
            content: The AI response content

        Returns:
            List of parsed edit instructions

        Raises:
            MalformedEditError: If content cannot be parsed
        """
        pass

    @abstractmethod
    def apply_edit(self, instruction: EditInstruction, original_content: str) -> EditResult:
        """
        Apply a single edit instruction to content.

        Args:
            instruction: The edit instruction
            original_content: Original file content

        Returns:
            EditResult with the outcome
        """
        pass

    def pre_process_content(self, content: str) -> str:
        """Pre-process content before applying edits."""
        return content

    def post_process_content(self, content: str) -> str:
        """Post-process content after applying edits."""
        return content

    def get_error_suggestions(self, error: Exception) -> List[str]:
        """Get recovery suggestions for errors."""
        return [
            "Verify the edit format is correct",
            "Check file paths and content",
            "Try breaking complex edits into smaller parts"
        ]


# =============================================================================
# Concrete Strategy Implementations
# =============================================================================

class SearchReplaceStrategy(EditStrategy):
    """Strategy for search/replace block edits."""

    @property
    def edit_format(self) -> str:
        return "search-replace"

    @property
    def supported_operations(self) -> List[str]:
        return ["create", "update", "delete"]

    def validate_edit_instruction(self, instruction: EditInstruction) -> bool:
        """Validate search/replace instruction format."""
        if not instruction.content:
            raise ValidationError("Empty edit content")

        # Check for required markers
        if "<<<<<<< SEARCH" not in instruction.content:
            raise MalformedEditError(
                instruction.content,
                "search-replace",
                "Missing '<<<<<<< SEARCH' marker"
            )

        if "=======" not in instruction.content:
            raise MalformedEditError(
                instruction.content,
                "search-replace",
                "Missing '=======' divider"
            )

        if ">>>>>>> REPLACE" not in instruction.content:
            raise MalformedEditError(
                instruction.content,
                "search-replace",
                "Missing '>>>>>>> REPLACE' marker"
            )

        return True

    def parse_edits(self, content: str) -> List[EditInstruction]:
        """Parse search/replace blocks from content."""
        instructions = []

        # Pattern to match fenced code blocks with file paths
        pattern = r'```\w*\n([^\n]+)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE\n```'

        matches = re.findall(pattern, content, re.DOTALL)

        for file_path, search_text, replace_text in matches:
            edit_content = f"<<<<<<< SEARCH\n{search_text}\n=======\n{replace_text}\n>>>>>>> REPLACE"

            instructions.append(EditInstruction(
                file_path=file_path.strip(),
                edit_type="search_replace",
                content=edit_content,
                metadata={"search_text": search_text, "replace_text": replace_text}
            ))

        return instructions

    @handle_edit_errors
    def apply_edit(self, instruction: EditInstruction, original_content: str) -> EditResult:
        """Apply search/replace edit."""
        try:
            self.validate_edit_instruction(instruction)

            # Extract search and replace text
            lines = instruction.content.strip().split('\n')
            search_start = lines.index("<<<<<<< SEARCH") + 1
            divider = lines.index("=======")
            replace_end = lines.index(">>>>>>> REPLACE")

            search_text = '\n'.join(lines[search_start:divider])
            replace_text = '\n'.join(lines[divider + 1:replace_end])

            # Handle empty search (new file creation)
            if not search_text.strip():
                if not original_content:
                    return EditResult(
                        success=True,
                        file_path=instruction.file_path,
                        original_content=original_content,
                        new_content=replace_text
                    )
                else:
                    # Append to existing file
                    new_content = original_content + replace_text
                    return EditResult(
                        success=True,
                        file_path=instruction.file_path,
                        original_content=original_content,
                        new_content=new_content
                    )

            # Count occurrences
            search_count = original_content.count(search_text)

            if search_count == 0:
                raise SearchTextNotFoundError(search_text, instruction.file_path)
            elif search_count > 1:
                raise SearchTextNotUniqueError(search_text, instruction.file_path, search_count)

            # Perform replacement
            new_content = original_content.replace(search_text, replace_text)

            return EditResult(
                success=True,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=new_content
            )

        except Exception as e:
            return EditResult(
                success=False,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=original_content,
                error_message=str(e)
            )


class UnifiedDiffStrategy(EditStrategy):
    """Strategy for unified diff edits."""

    @property
    def edit_format(self) -> str:
        return "unified-diff"

    @property
    def supported_operations(self) -> List[str]:
        return ["create", "update", "delete"]

    def validate_edit_instruction(self, instruction: EditInstruction) -> bool:
        """Validate unified diff format."""
        content = instruction.content.strip()

        if not content.startswith("---") or "+++" not in content:
            raise MalformedEditError(
                instruction.content,
                "unified-diff",
                "Missing diff headers (--- and +++)"
            )

        if "@@ " not in content:
            raise MalformedEditError(
                instruction.content,
                "unified-diff",
                "Missing hunk header (@@)"
            )

        return True

    def parse_edits(self, content: str) -> List[EditInstruction]:
        """Parse unified diff blocks from content."""
        instructions = []

        # Find diff blocks
        diff_pattern = r'```diff\n(.*?)\n```'
        matches = re.findall(diff_pattern, content, re.DOTALL)

        for diff_content in matches:
            lines = diff_content.split('\n')

            # Extract file path from diff headers
            file_path = None
            for line in lines:
                if line.startswith("+++ "):
                    file_path = line[4:].strip()
                    # Remove a/ or b/ prefixes if present
                    if file_path.startswith("b/"):
                        file_path = file_path[2:]
                    break

            if file_path:
                instructions.append(EditInstruction(
                    file_path=file_path,
                    edit_type="unified_diff",
                    content=diff_content
                ))

        return instructions

    @handle_edit_errors
    def apply_edit(self, instruction: EditInstruction, original_content: str) -> EditResult:
        """Apply unified diff edit."""
        try:
            self.validate_edit_instruction(instruction)

            # Parse diff and apply changes
            diff_lines = instruction.content.split('\n')

            # Extract hunks and apply them
            new_content = self._apply_diff_hunks(original_content, diff_lines)

            return EditResult(
                success=True,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=new_content
            )

        except Exception as e:
            return EditResult(
                success=False,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=original_content,
                error_message=str(e)
            )

    def _apply_diff_hunks(self, content: str, diff_lines: List[str]) -> str:
        """Apply diff hunks to content."""
        # This is a simplified implementation
        # In practice, you'd want a more robust diff application algorithm

        content_lines = content.splitlines(keepends=True)
        result_lines = content_lines.copy()

        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]

            if line.startswith("@@"):
                # Parse hunk header and apply changes
                i = self._apply_hunk(result_lines, diff_lines, i)
            else:
                i += 1

        return ''.join(result_lines)

    def _apply_hunk(self, content_lines: List[str], diff_lines: List[str], start_idx: int) -> int:
        """Apply a single hunk to the content."""
        # Simplified hunk application - would need more robust implementation
        i = start_idx + 1

        removals = []
        additions = []

        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            line = diff_lines[i]
            if line.startswith("-"):
                removals.append(line[1:])
            elif line.startswith("+"):
                additions.append(line[1:])
            i += 1

        # Apply the changes (simplified)
        for removal in removals:
            for j, content_line in enumerate(content_lines):
                if content_line.strip() == removal.strip():
                    content_lines[j] = ""
                    break

        # Add new lines where appropriate
        if additions:
            content_lines.extend(additions)

        return i


class WholeFileStrategy(EditStrategy):
    """Strategy for whole file replacement edits."""

    @property
    def edit_format(self) -> str:
        return "whole-file"

    @property
    def supported_operations(self) -> List[str]:
        return ["create", "update"]

    def validate_edit_instruction(self, instruction: EditInstruction) -> bool:
        """Validate whole file instruction."""
        if not instruction.content:
            raise ValidationError("Empty file content")
        return True

    def parse_edits(self, content: str) -> List[EditInstruction]:
        """Parse whole file blocks from content."""
        instructions = []

        # Pattern to match file blocks
        lines = content.split('\n')
        current_file = None
        current_content = []
        in_code_block = False

        for line in lines:
            if line.startswith('```') and not in_code_block:
                # Start of code block - previous line should be filename
                in_code_block = True
                current_content = []
            elif line.startswith('```') and in_code_block:
                # End of code block
                if current_file:
                    instructions.append(EditInstruction(
                        file_path=current_file,
                        edit_type="whole_file",
                        content='\n'.join(current_content)
                    ))
                in_code_block = False
                current_file = None
                current_content = []
            elif in_code_block:
                current_content.append(line)
            elif not in_code_block and line.strip() and not line.startswith(' '):
                # Potential filename
                stripped = line.strip().rstrip(':').strip('*').strip('`').strip('#').strip()
                if stripped and '/' in stripped or '.' in stripped:
                    current_file = stripped

        return instructions

    @handle_edit_errors
    def apply_edit(self, instruction: EditInstruction, original_content: str) -> EditResult:
        """Apply whole file replacement."""
        try:
            self.validate_edit_instruction(instruction)

            return EditResult(
                success=True,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=instruction.content
            )

        except Exception as e:
            return EditResult(
                success=False,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=original_content,
                error_message=str(e)
            )


class PatchStrategy(EditStrategy):
    """Strategy for patch format edits."""

    @property
    def edit_format(self) -> str:
        return "patch"

    @property
    def supported_operations(self) -> List[str]:
        return ["create", "update", "delete"]

    def validate_edit_instruction(self, instruction: EditInstruction) -> bool:
        """Validate patch instruction format."""
        content = instruction.content.strip()

        if not content.startswith("*** Begin Patch"):
            raise MalformedEditError(
                instruction.content,
                "patch",
                "Missing '*** Begin Patch' marker"
            )

        if not content.endswith("*** End Patch"):
            raise MalformedEditError(
                instruction.content,
                "patch",
                "Missing '*** End Patch' marker"
            )

        return True

    def parse_edits(self, content: str) -> List[EditInstruction]:
        """Parse patch format blocks from content."""
        instructions = []

        # Find patch blocks
        patch_pattern = r'\*\*\* Begin Patch(.*?)\*\*\* End Patch'
        matches = re.findall(patch_pattern, content, re.DOTALL)

        for patch_content in matches:
            # Parse individual file operations within the patch
            file_ops = self._parse_file_operations(patch_content)
            instructions.extend(file_ops)

        return instructions

    def _parse_file_operations(self, patch_content: str) -> List[EditInstruction]:
        """Parse individual file operations from patch content."""
        instructions = []
        lines = patch_content.strip().split('\n')

        current_file = None
        current_action = None
        current_content = []

        for line in lines:
            if line.startswith("*** "):
                # Save previous operation
                if current_file and current_action:
                    instructions.append(EditInstruction(
                        file_path=current_file,
                        edit_type=current_action.lower(),
                        content='\n'.join(current_content),
                        metadata={"action": current_action}
                    ))

                # Parse new operation
                parts = line.split(" ", 3)
                if len(parts) >= 3:
                    current_action = parts[1]  # Add, Update, Delete
                    current_file = parts[3] if len(parts) > 3 else parts[2]
                    current_content = []
            else:
                current_content.append(line)

        # Save final operation
        if current_file and current_action:
            instructions.append(EditInstruction(
                file_path=current_file,
                edit_type=current_action.lower(),
                content='\n'.join(current_content),
                metadata={"action": current_action}
            ))

        return instructions

    @handle_edit_errors
    def apply_edit(self, instruction: EditInstruction, original_content: str) -> EditResult:
        """Apply patch format edit."""
        try:
            self.validate_edit_instruction(instruction)

            action = instruction.metadata.get("action", "").lower()

            if action == "add":
                # New file creation
                new_content = instruction.content
            elif action == "delete":
                # File deletion
                new_content = ""
            elif action == "update":
                # Apply patch-style updates
                new_content = self._apply_patch_updates(original_content, instruction.content)
            else:
                raise EditOperationError(f"Unknown patch action: {action}")

            return EditResult(
                success=True,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=new_content,
                metadata={"action": action}
            )

        except Exception as e:
            return EditResult(
                success=False,
                file_path=instruction.file_path,
                original_content=original_content,
                new_content=original_content,
                error_message=str(e)
            )

    def _apply_patch_updates(self, original_content: str, patch_content: str) -> str:
        """Apply patch-style updates to content."""
        # Simplified patch application
        # In practice, this would need more sophisticated diff logic

        orig_lines = original_content.splitlines()
        patch_lines = patch_content.splitlines()

        result_lines = []
        i = 0

        for patch_line in patch_lines:
            if patch_line.startswith("+"):
                result_lines.append(patch_line[1:])
            elif patch_line.startswith("-"):
                # Skip removed lines
                continue
            elif patch_line.startswith(" "):
                result_lines.append(patch_line[1:])
            elif patch_line.startswith("@@"):
                # Context marker - continue processing
                continue

        return '\n'.join(result_lines)


class NoOpStrategy(EditStrategy):
    """Strategy for read-only operations (ask, help, context)."""

    @property
    def edit_format(self) -> str:
        return "no-op"

    @property
    def supported_operations(self) -> List[str]:
        return []  # No edit operations

    def validate_edit_instruction(self, instruction: EditInstruction) -> bool:
        """No validation needed for no-op strategy."""
        return True

    def parse_edits(self, content: str) -> List[EditInstruction]:
        """No edits to parse for no-op strategy."""
        return []

    def apply_edit(self, instruction: EditInstruction, original_content: str) -> EditResult:
        """No edits to apply for no-op strategy."""
        return EditResult(
            success=True,
            file_path=instruction.file_path,
            original_content=original_content,
            new_content=original_content
        )


# =============================================================================
# Strategy Factory
# =============================================================================

class EditStrategyFactory:
    """
    Factory for creating appropriate edit strategies based on configuration.

    This implements the Factory pattern to encapsulate strategy selection logic
    and make it easy to add new strategies without modifying existing code.
    """

    _strategies = {
        "diff": SearchReplaceStrategy,
        "diff-fenced": SearchReplaceStrategy,
        "editor-diff": SearchReplaceStrategy,
        "editor-diff-fenced": SearchReplaceStrategy,
        "udiff": UnifiedDiffStrategy,
        "udiff-simple": UnifiedDiffStrategy,
        "patch": PatchStrategy,
        "whole": WholeFileStrategy,
        "editor-whole": WholeFileStrategy,
        "func": WholeFileStrategy,
        "ask": NoOpStrategy,
        "help": NoOpStrategy,
        "context": NoOpStrategy,
        "architect": NoOpStrategy,
    }

    @classmethod
    def create_strategy(
        cls,
        edit_format: str,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        file_size_kb: Optional[int] = None
    ) -> EditStrategy:
        """
        Create the appropriate edit strategy.

        Args:
            edit_format: The edit format identifier
            config: Optional configuration
            model_name: Optional model name for optimization
            file_size_kb: Optional file size for optimization

        Returns:
            Configured edit strategy

        Raises:
            UnknownEditFormat: If format is not supported
        """
        if edit_format not in cls._strategies:
            from .exceptions import UnknownEditFormat
            raise UnknownEditFormat(edit_format, list(cls._strategies.keys()))

        strategy_class = cls._strategies[edit_format]

        # Enhance config based on context
        enhanced_config = config or {}
        enhanced_config.update({
            "edit_format": edit_format,
            "model_name": model_name,
            "file_size_kb": file_size_kb
        })

        # Apply optimization heuristics
        enhanced_config = cls._optimize_config(enhanced_config)

        return strategy_class(enhanced_config)

    @classmethod
    def _optimize_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization heuristics based on context."""
        model_name = config.get("model_name", "")
        file_size_kb = config.get("file_size_kb", 0)

        # Optimize based on model capabilities
        if "gpt-4" in model_name.lower():
            config["prefer_precise_diffs"] = True
            config["max_context_lines"] = 10
        elif "claude" in model_name.lower():
            config["prefer_search_replace"] = True
            config["max_context_lines"] = 5

        # Optimize based on file size
        if file_size_kb and file_size_kb > 50:
            config["prefer_targeted_edits"] = True
            config["chunk_large_files"] = True

        return config

    @classmethod
    def register_strategy(cls, edit_format: str, strategy_class: type):
        """Register a new edit strategy."""
        if not issubclass(strategy_class, EditStrategy):
            raise ValueError("Strategy class must inherit from EditStrategy")

        cls._strategies[edit_format] = strategy_class

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported edit formats."""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, edit_format: str) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        if edit_format not in cls._strategies:
            raise ValueError(f"Unknown edit format: {edit_format}")

        strategy_class = cls._strategies[edit_format]
        temp_strategy = strategy_class()

        return {
            "name": strategy_class.__name__,
            "edit_format": temp_strategy.edit_format,
            "supported_operations": temp_strategy.supported_operations,
            "description": strategy_class.__doc__ or "No description available"
        }


# =============================================================================
# Strategy Coordinator
# =============================================================================

class EditStrategyCoordinator:
    """
    Coordinates multiple edit strategies and handles complex editing workflows.

    This class provides a high-level interface for managing edits across
    different formats and handles error recovery, validation, and optimization.
    """

    def __init__(self, default_strategy: Optional[EditStrategy] = None):
        """
        Initialize the coordinator.

        Args:
            default_strategy: Default strategy to use if none specified
        """
        self.default_strategy = default_strategy
        self.recovery_strategies = []
        self.validation_enabled = True
        self.dry_run_mode = False

    def add_recovery_strategy(self, recovery_strategy):
        """Add an error recovery strategy."""
        self.recovery_strategies.append(recovery_strategy)

    def set_validation_enabled(self, enabled: bool):
        """Enable or disable validation."""
        self.validation_enabled = enabled

    def set_dry_run_mode(self, enabled: bool):
        """Enable or disable dry run mode."""
        self.dry_run_mode = enabled

    def process_edits(
        self,
        content: str,
        strategy: Optional[EditStrategy] = None,
        file_contents: Optional[Dict[str, str]] = None
    ) -> List[EditResult]:
        """
        Process edits using the specified or default strategy.

        Args:
            content: AI response content containing edits
            strategy: Optional specific strategy to use
            file_contents: Optional dict of file_path -> content

        Returns:
            List of edit results
        """
        edit_strategy = strategy or self.default_strategy
        if not edit_strategy:
            raise ConfigurationError("No edit strategy specified")

        try:
            # Parse edit instructions
            instructions = edit_strategy.parse_edits(content)

            if not instructions:
                return []

            # Apply edits
            results = []
            for instruction in instructions:
                if self.validation_enabled:
                    edit_strategy.validate_edit_instruction(instruction)

                # Get original content
                original_content = ""
                if file_contents and instruction.file_path in file_contents:
                    original_content = file_contents[instruction.file_path]
                elif Path(instruction.file_path).exists():
                    original_content = Path(instruction.file_path).read_text()

                # Apply edit
                if self.dry_run_mode:
                    result = EditResult(
                        success=True,
                        file_path=instruction.file_path,
                        original_content=original_content,
                        new_content="[DRY RUN - NOT APPLIED]",
                        metadata={"dry_run": True}
                    )
                else:
                    result = edit_strategy.apply_edit(instruction, original_content)

                results.append(result)

            return results

        except Exception as e:
            # Try recovery strategies
            for recovery in self.recovery_strategies:
                if recovery.can_recover(e):
                    try:
                        return recovery.recover(e, {
                            "content": content,
                            "strategy": edit_strategy,
                            "file_contents": file_contents
                        })
                    except Exception:
                        continue  # Try next recovery strategy

            # No recovery possible, re-raise
            raise

    def validate_results(self, results: List[EditResult]) -> Tuple[List[EditResult], List[EditResult]]:
        """
        Validate edit results and separate successful from failed edits.

        Args:
            results: List of edit results to validate

        Returns:
            Tuple of (successful_results, failed_results)
        """
        successful = []
        failed = []

        for result in results:
            if result.success and not result.error_message:
                successful.append(result)
            else:
                failed.append(result)

        return successful, failed

    def get_edit_summary(self, results: List[EditResult]) -> Dict[str, Any]:
        """Get a summary of edit operations."""
        successful, failed = self.validate_results(results)

        return {
            "total_edits": len(results),
            "successful_edits": len(successful),
            "failed_edits": len(failed),
            "files_modified": [r.file_path for r in successful],
            "files_failed": [r.file_path for r in failed],
            "errors": [r.error_message for r in failed if r.error_message]
        }


# =============================================================================
# Strategy Selection Helpers
# =============================================================================

def select_optimal_strategy(
    model_name: str,
    file_size_kb: int,
    edit_complexity: str = "medium",
    user_preference: Optional[str] = None
) -> str:
    """
    Select the optimal edit strategy based on context.

    Args:
        model_name: Name of the AI model being used
        file_size_kb: Size of the file being edited
        edit_complexity: Complexity level (simple, medium, complex)
        user_preference: User's preferred edit format

    Returns:
        Recommended edit format string
    """
    # Honor user preference if valid
    if user_preference and user_preference in EditStrategyFactory.get_supported_formats():
        return user_preference

    # Model-based optimization
    model_lower = model_name.lower()

    # Large files benefit from targeted edits
    if file_size_kb > 100:
        if "gpt-4" in model_lower or "claude" in model_lower:
            return "udiff"  # Precise diffs for large files
        else:
            return "diff"  # Search/replace for other models

    # Complex edits benefit from more structured formats
    if edit_complexity == "complex":
        if "gpt-4-turbo" in model_lower or "claude-3" in model_lower:
            return "patch"  # Most structured format
        else:
            return "udiff"  # Good middle ground

    # Simple edits can use basic formats
    if edit_complexity == "simple":
        if file_size_kb < 10:
            return "whole"  # Just replace the whole small file
        else:
            return "diff"  # Simple search/replace

    # Default recommendations for medium complexity
    if "gpt-4" in model_lower:
        return "udiff"  # GPT-4 is good with diffs
    elif "claude" in model_lower:
        return "diff-fenced"  # Claude prefers fenced blocks
    elif "gemini" in model_lower:
        return "diff"  # Gemini works well with search/replace
    else:
        return "diff"  # Safe default


def get_strategy_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Get recommendations for when to use each strategy.

    Returns:
        Dictionary mapping strategy names to recommendation info
    """
    return {
        "diff": {
            "best_for": ["Small to medium files", "Simple edits", "Most models"],
            "pros": ["Clear format", "Good model support", "Easy to understand"],
            "cons": ["Can struggle with complex edits", "Requires exact matches"],
            "recommended_models": ["gpt-3.5", "gemini", "claude-instant"]
        },
        "diff-fenced": {
            "best_for": ["Claude models", "Complex search/replace", "Multiple edits"],
            "pros": ["Better context", "Claude optimized", "Handles multiple files"],
            "cons": ["More verbose", "Requires fence support"],
            "recommended_models": ["claude-3", "claude-2"]
        },
        "udiff": {
            "best_for": ["Large files", "Precise edits", "Advanced models"],
            "pros": ["Very precise", "Handles complex changes", "Git-compatible"],
            "cons": ["Complex format", "Requires model sophistication"],
            "recommended_models": ["gpt-4", "gpt-4-turbo", "claude-3-opus"]
        },
        "patch": {
            "best_for": ["Multiple file operations", "Complex workflows", "Batch edits"],
            "pros": ["Handles multiple files", "Structured format", "Clear operations"],
            "cons": ["Most complex", "Requires advanced models"],
            "recommended_models": ["gpt-4-turbo", "claude-3-opus"]
        },
        "whole": {
            "best_for": ["Small files", "Complete rewrites", "New files"],
            "pros": ["Simple concept", "Always complete", "Good for small files"],
            "cons": ["Token intensive", "Not efficient for large files"],
            "recommended_models": ["Any model"]
        }
    }


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_edit_content(content: str, edit_format: str) -> List[str]:
    """
    Validate edit content and return list of issues found.

    Args:
        content: The edit content to validate
        edit_format: The expected edit format

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    if not content or not content.strip():
        issues.append("Edit content is empty")
        return issues

    # Format-specific validation
    if edit_format in ["diff", "diff-fenced", "editor-diff", "editor-diff-fenced"]:
        issues.extend(_validate_search_replace_content(content))
    elif edit_format in ["udiff", "udiff-simple"]:
        issues.extend(_validate_unified_diff_content(content))
    elif edit_format == "patch":
        issues.extend(_validate_patch_content(content))
    elif edit_format in ["whole", "editor-whole"]:
        issues.extend(_validate_whole_file_content(content))

    return issues


def _validate_search_replace_content(content: str) -> List[str]:
    """Validate search/replace format content."""
    issues = []

    # Check for required markers
    if "<<<<<<< SEARCH" not in content:
        issues.append("Missing '<<<<<<< SEARCH' marker")

    if "=======" not in content:
        issues.append("Missing '=======' divider")

    if ">>>>>>> REPLACE" not in content:
        issues.append("Missing '>>>>>>> REPLACE' marker")

    # Check for balanced blocks
    search_count = content.count("<<<<<<< SEARCH")
    replace_count = content.count(">>>>>>> REPLACE")
    if search_count != replace_count:
        issues.append(f"Unbalanced search/replace blocks: {search_count} SEARCH, {replace_count} REPLACE")

    return issues


def _validate_unified_diff_content(content: str) -> List[str]:
    """Validate unified diff format content."""
    issues = []

    if "```diff" not in content:
        issues.append("Missing diff code block")

    # Check for diff headers
    if "--- " not in content or "+++ " not in content:
        issues.append("Missing diff file headers (--- and +++)")

    # Check for hunk markers
    if "@@ " not in content:
        issues.append("Missing hunk headers (@@)")

    return issues


def _validate_patch_content(content: str) -> List[str]:
    """Validate patch format content."""
    issues = []

    if "*** Begin Patch" not in content:
        issues.append("Missing '*** Begin Patch' marker")

    if "*** End Patch" not in content:
        issues.append("Missing '*** End Patch' marker")

    # Check for file operations
    if not any(marker in content for marker in ["*** Add File:", "*** Update File:", "*** Delete File:"]):
        issues.append("No file operations found in patch")

    return issues


def _validate_whole_file_content(content: str) -> List[str]:
    """Validate whole file format content."""
    issues = []

    # Check for code blocks
    if "```" not in content:
        issues.append("No code blocks found")

    # Check for balanced code blocks
    fence_count = content.count("```")
    if fence_count % 2 != 0:
        issues.append("Unbalanced code block fences")

    return issues


# =============================================================================
# Performance and Metrics
# =============================================================================

@dataclass
class EditMetrics:
    """Metrics for edit operations."""
    strategy_used: str
    files_processed: int
    successful_edits: int
    failed_edits: int
    total_time_ms: float
    tokens_used: Optional[int] = None
    lines_added: int = 0
    lines_removed: int = 0
    lines_modified: int = 0


class PerformanceTracker:
    """Track performance metrics for edit strategies."""

    def __init__(self):
        self.metrics: List[EditMetrics] = []

    def record_edit_session(self, metrics: EditMetrics):
        """Record metrics from an edit session."""
        self.metrics.append(metrics)

    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific strategy."""
        strategy_metrics = [m for m in self.metrics if m.strategy_used == strategy_name]

        if not strategy_metrics:
            return {"error": f"No metrics found for strategy: {strategy_name}"}

        total_files = sum(m.files_processed for m in strategy_metrics)
        total_successful = sum(m.successful_edits for m in strategy_metrics)
        total_failed = sum(m.failed_edits for m in strategy_metrics)
        avg_time = sum(m.total_time_ms for m in strategy_metrics) / len(strategy_metrics)

        success_rate = (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0

        return {
            "strategy": strategy_name,
            "sessions": len(strategy_metrics),
            "total_files_processed": total_files,
            "total_successful_edits": total_successful,
            "total_failed_edits": total_failed,
            "success_rate_percent": round(success_rate, 2),
            "average_time_ms": round(avg_time, 2)
        }

    def get_overall_performance(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics:
            return {"error": "No metrics recorded"}

        strategies = set(m.strategy_used for m in self.metrics)
        strategy_performance = {
            strategy: self.get_strategy_performance(strategy)
            for strategy in strategies
        }

        total_sessions = len(self.metrics)
        total_files = sum(m.files_processed for m in self.metrics)
        total_time = sum(m.total_time_ms for m in self.metrics)

        return {
            "total_sessions": total_sessions,
            "total_files_processed": total_files,
            "total_time_ms": round(total_time, 2),
            "strategies_used": list(strategies),
            "strategy_breakdown": strategy_performance
        }
