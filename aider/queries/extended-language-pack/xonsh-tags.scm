; Xonsh language tree-sitter tags query file
; Language type: shell
; Extensions: .xsh

; Function definitions
(function_def
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call

; Variable definitions
(assignment
+  name: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

; Command invocations
(command
+  name: (command_name) @name.reference.command) @reference.command

; Environment variables
(variable_expansion
+  name: (variable_name) @name.reference.env_var) @reference.env_var

; Command aliases
(alias_declaration
+  name: (identifier) @name.definition.alias) @definition.alias

; Pipeline operations
(pipeline
+  (command
+    name: (command_name) @name.reference.pipe_command) @reference.command

