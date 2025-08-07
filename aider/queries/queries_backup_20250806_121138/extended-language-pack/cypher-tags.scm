; Cypher language tree-sitter tags query file
; Language type: database
; Extensions: .cypher, .cql


; Function definitions
(function_invocation
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Table references
(table_reference
+  name: (identifier) @name.reference.table) @reference.table

; Column references
(column_reference
+  name: (identifier) @name.reference.column) @reference.column

; Database functions
(function_call
+  name: (identifier) @name.reference.database_function) @reference.database_function

; Query operations
(query_expression
+  (identifier) @name.reference.query_op) @reference.query_op

