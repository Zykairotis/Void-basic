; Tree-sitter query file for Triton
; Language: Triton
; Type: Ml
; Description: GPU kernel programming
; Version: 1.0
; Generated: 2025-08-06
; Features: kernels, blocks, grids, memory


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

; Kernel definitions
(kernel_definition
  name: (identifier) @name.definition.kernel) @definition.kernel

; Model definitions
(model_definition
  name: (identifier) @name.definition.model) @definition.model

; Layer definitions
(layer_definition
  name: (identifier) @name.definition.layer) @definition.layer

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
