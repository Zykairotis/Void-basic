; Tla language tree-sitter tags query file
; Language type: specification
; Extensions: .tla


; Module definitions
(module
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module


; Variable definitions
(variable_declaration
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

