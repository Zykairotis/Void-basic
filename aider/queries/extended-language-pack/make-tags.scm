; Make language tree-sitter tags query file
; Language type: build
; Extensions: Makefile, .mk

; Variable definitions
(variable
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

