; Vhdl language tree-sitter tags query file
; Language type: hardware
; Extensions: .vhd, .vhdl


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

