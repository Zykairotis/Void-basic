; Tree-sitter query file for Kubernetes
; Language: Kubernetes
; Type: Orchestration
; Description: Container orchestration
; Version: 1.0
; Generated: 2025-08-06
; Features: resources, specs, metadata, selectors


; Variable definitions
(variable_declaration
  name: (identifier) @name.definition.variable) @definition.variable

(assignment_expression
  left: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

; Type definitions
(type_definition
  name: (identifier) @name.definition.type) @definition.type

(type_declaration
  name: (identifier) @name.definition.type) @definition.type

; Language-specific constructs

; Metadata patterns
(metadata
  name: (identifier) @name.definition.metadata) @definition.metadata

(metadata
  (identifier) @name.reference.metadata) @reference.metadata

; Resource definitions
(resource_definition
  type: (identifier) @name.definition.resource) @definition.resource

; Module references
(module_reference
  source: (string) @name.reference.module) @reference.module

; Variable references
(variable_reference
  name: (identifier) @name.reference.variable) @reference.variable

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
