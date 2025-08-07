; Verilog language tree-sitter tags query file
; Language type: hardware
; Extensions: .v, .vh


; Module definitions
(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Module references
(import_statement
+  (identifier) @name.reference.module) @reference.module


; Hardware modules
(module_declaration
+  name: (identifier) @name.definition.module) @definition.module

; Port declarations
(port_declaration
+  name: (identifier) @name.definition.port) @definition.port

; Signal declarations
(signal_declaration
+  name: (identifier) @name.definition.signal) @definition.signal

; Wire connections
(wire_assignment
+  target: (identifier) @name.reference.wire) @reference.wire

