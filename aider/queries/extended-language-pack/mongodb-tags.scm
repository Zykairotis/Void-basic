; Mongodb language tree-sitter tags query file
; Language type: database
; Extensions: .js

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

