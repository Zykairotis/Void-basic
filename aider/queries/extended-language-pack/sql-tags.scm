; SQL database language tree-sitter tags query file

; Table definitions
(create_table_statement
  name: (identifier) @name.definition.table) @definition.table

(create_table_statement
  schema: (identifier) @name.reference.schema
  name: (identifier) @name.definition.table) @definition.table

; Column definitions
(column_definition
  name: (identifier) @name.definition.column) @definition.column

; Primary key constraints
(primary_key_constraint
  (identifier) @name.reference.column) @reference.column

; Foreign key constraints
(foreign_key_constraint
  columns: (identifier) @name.reference.column
  references: (table_reference
    name: (identifier) @name.reference.table) @reference.table

; Index definitions
(create_index_statement
  name: (identifier) @name.definition.index
  table: (identifier) @name.reference.table) @definition.index

; View definitions
(create_view_statement
  name: (identifier) @name.definition.view) @definition.view

; Stored procedure definitions
(create_procedure_statement
  name: (identifier) @name.definition.procedure) @definition.procedure

(create_function_statement
  name: (identifier) @name.definition.function) @definition.function

; Trigger definitions
(create_trigger_statement
  name: (identifier) @name.definition.trigger
  table: (identifier) @name.reference.table) @definition.trigger

; Database/Schema definitions
(create_database_statement
  name: (identifier) @name.definition.database) @definition.database

(create_schema_statement
  name: (identifier) @name.definition.schema) @definition.schema

; Table references in queries
(table_reference
  name: (identifier) @name.reference.table) @reference.table

(table_reference
  schema: (identifier) @name.reference.schema
  name: (identifier) @name.reference.table) @reference.table

; Column references
(column_reference
  name: (identifier) @name.reference.column) @reference.column

(column_reference
  table: (identifier) @name.reference.table
  name: (identifier) @name.reference.column) @reference.column

; Qualified column references
(qualified_column_reference
  schema: (identifier) @name.reference.schema
  table: (identifier) @name.reference.table
  column: (identifier) @name.reference.column) @reference.column

; Table aliases
(table_alias
  table: (identifier) @name.reference.table
  alias: (identifier) @name.definition.alias) @definition.alias

; Column aliases
(column_alias
  expression: (column_reference
    name: (identifier) @name.reference.column
  alias: (identifier) @name.definition.alias) @definition.alias

; Function calls
(function_call
  name: (identifier) @name.reference.call) @reference.call

; Aggregate functions
(function_call
  name: (identifier) @name.reference.aggregate
  (#any-of? @name.reference.aggregate)
    "COUNT" "SUM" "AVG" "MIN" "MAX" "GROUP_CONCAT"
    "STRING_AGG" "ARRAY_AGG" "COLLECT" "LISTAGG") @reference.aggregate

; Window functions
(window_function
  function: (function_call
    name: (identifier) @name.reference.window_function) @reference.window_function

; Built-in functions
(function_call
  name: (identifier) @name.reference.builtin
  (#any-of? @name.reference.builtin)
    "COALESCE" "NULLIF" "CASE" "CAST" "CONVERT" "SUBSTRING"
    "LEFT" "RIGHT" "LENGTH" "LEN" "TRIM" "UPPER" "LOWER"
    "CONCAT" "REPLACE" "REGEXP_REPLACE" "TO_CHAR" "TO_DATE"
    "DATE_FORMAT" "NOW" "CURRENT_TIMESTAMP" "GETDATE"
    "EXTRACT" "DATE_PART" "DATEDIFF" "DATEADD" "ABS" "ROUND"
    "FLOOR" "CEIL" "CEILING" "POWER" "SQRT" "RANDOM" "RAND") @reference.builtin

; Variables (for stored procedures/functions)
(variable_declaration
  name: (identifier) @name.definition.variable) @definition.variable

(variable_reference
  name: (identifier) @name.reference.variable) @reference.variable

; Parameters in stored procedures
(parameter_declaration
  name: (identifier) @name.definition.parameter) @definition.parameter

; Common Table Expressions (CTEs)
(with_clause
  (cte_definition
    name: (identifier) @name.definition.cte) @definition.cte

(table_reference
  name: (identifier) @name.reference.cte) @reference.cte

; Subquery aliases
(derived_table
  subquery: (select_statement)
  alias: (identifier) @name.definition.subquery_alias) @definition.alias

; Join conditions
(join_clause
  table: (table_reference
    name: (identifier) @name.reference.join_table) @reference.table

; Union operations
(union_statement
  left: (select_statement)
  right: (select_statement) @reference.union

; Insert statements
(insert_statement
  table: (table_reference
    name: (identifier) @name.reference.table) @reference.table

(insert_statement
  columns: (column_list
    (identifier) @name.reference.column) @reference.column

; Update statements
(update_statement
  table: (table_reference
    name: (identifier) @name.reference.table) @reference.table

(update_statement
  set_clause: (assignment
    column: (identifier) @name.reference.column) @reference.column

; Delete statements
(delete_statement
  table: (table_reference
    name: (identifier) @name.reference.table) @reference.table

; Constraint names
(constraint_definition
  name: (identifier) @name.definition.constraint) @definition.constraint

; Index columns
(index_column
  name: (identifier) @name.reference.column) @reference.column

; Data types
(data_type
  name: (identifier) @name.reference.type) @reference.type

; User-defined types
(create_type_statement
  name: (identifier) @name.definition.type) @definition.type

; Sequences
(create_sequence_statement
  name: (identifier) @name.definition.sequence) @definition.sequence

(sequence_reference
  name: (identifier) @name.reference.sequence) @reference.sequence

; Temporary tables
(create_table_statement
  temporary: "TEMPORARY"
  name: (identifier) @name.definition.temp_table) @definition.temp_table

; Materialized views
(create_materialized_view_statement
  name: (identifier) @name.definition.materialized_view) @definition.materialized_view

; Partitions
(partition_clause
  name: (identifier) @name.definition.partition) @definition.partition

; Grant/Revoke statements
(grant_statement
  object: (identifier) @name.reference.grant_object) @reference.grant_object

(revoke_statement
  object: (identifier) @name.reference.revoke_object) @reference.revoke_object

; Role definitions
(create_role_statement
  name: (identifier) @name.definition.role) @definition.role

; User definitions
(create_user_statement
  name: (identifier) @name.definition.user) @definition.user

; Cursor definitions
(cursor_declaration
  name: (identifier) @name.definition.cursor) @definition.cursor

; Exception handling
(exception_handler
  name: (identifier) @name.definition.exception) @definition.exception

; Loop labels
(loop_statement
  label: (identifier) @name.definition.label) @definition.label

; Case expressions
(case_expression
  (when_clause
    condition: (column_reference
      name: (identifier) @name.reference.column)) @reference.column

; Order by clauses
(order_by_clause
  (order_expression
    expression: (column_reference
      name: (identifier) @name.reference.column)) @reference.column

; Group by clauses
(group_by_clause
  (column_reference
    name: (identifier) @name.reference.column) @reference.column

; Having clauses
(having_clause
  condition: (column_reference
    name: (identifier) @name.reference.column) @reference.column

; Where clauses
(where_clause
  condition: (column_reference
    name: (identifier) @name.reference.column) @reference.column

; Prepared statements
(prepare_statement
  name: (identifier) @name.definition.prepared_statement) @definition.prepared_statement

(execute_statement
  name: (identifier) @name.reference.prepared_statement) @reference.prepared_statement

; Database links
(database_link_reference
  name: (identifier) @name.reference.database_link) @reference.database_link

; Synonyms
(create_synonym_statement
  name: (identifier) @name.definition.synonym
  target: (identifier) @name.reference.synonym_target) @definition.synonym

; Packages (Oracle, PostgreSQL)
(create_package_statement
  name: (identifier) @name.definition.package) @definition.package

; Comments on database objects
(comment_statement
  object: (identifier) @name.reference.comment_object) @reference.comment_object

; Default values referencing other columns/functions
(default_clause
  value: (column_reference
    name: (identifier) @name.reference.default_column) @reference.column

(default_clause
  value: (function_call
    name: (identifier) @name.reference.default_function) @reference.call
