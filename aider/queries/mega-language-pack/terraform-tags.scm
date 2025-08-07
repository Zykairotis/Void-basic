; Tree-sitter query file for Terraform
; Language: Terraform
; Type: Infrastructure
; Description: Infrastructure as code
; Version: 1.0
; Generated: 2025-08-06
; Features: resources, variables, modules, outputs


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
