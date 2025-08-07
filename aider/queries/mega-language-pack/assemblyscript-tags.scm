; Tree-sitter query file for Assemblyscript
; Language: Assemblyscript
; Type: Web
; Description: TypeScript to WebAssembly compiler
; Version: 1.0
; Generated: 2025-08-06
; Features: classes, functions, types, decorators


; Class definitions
(class_definition
  name: (identifier) @name.definition.class) @definition.class

(class_declaration
  name: (identifier) @name.definition.class) @definition.class

; Method definitions
(method_definition
  name: (identifier) @name.definition.method) @definition.method

(method_declaration
  name: (identifier) @name.definition.method) @definition.method

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

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
