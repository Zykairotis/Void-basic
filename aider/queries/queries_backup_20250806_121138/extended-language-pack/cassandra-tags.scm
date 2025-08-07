; Cassandra language tree-sitter tags query file
; Language type: database
; Extensions: .cql


; Function definitions
(function_definition
+  name: (identifier) @name.definition.function) @definition.function

; Function calls
(call_expression
+  function: (identifier) @name.reference.call) @reference.call


; Type definitions
(type_definition
+  name: (identifier) @name.definition.type) @definition.type

; Type references
(type_identifier) @name.reference.type @reference.type


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

