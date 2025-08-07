; Coq language tree-sitter tags query file
; Language type: theorem_prover
; Extensions: .v

; Module definitions
(module
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module

