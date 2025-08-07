; Ada language tree-sitter tags query file

; Package definitions
(package_declaration
  name: (identifier) @name.definition.module) @definition.module

(package_body
  name: (identifier) @name.definition.module) @definition.module

; Procedure definitions
(procedure_declaration
  name: (identifier) @name.definition.function) @definition.function

(procedure_body
  name: (identifier) @name.definition.function) @definition.function

; Function definitions
(function_declaration
  name: (identifier) @name.definition.function) @definition.function

(function_body
  name: (identifier) @name.definition.function) @definition.function

; Type definitions
(type_declaration
  name: (identifier) @name.definition.type) @definition.type

(subtype_declaration
  name: (identifier) @name.definition.type) @definition.type

; Record type definitions
(record_type_definition
  (component_declaration
    name: (identifier) @name.definition.field)) @definition.field

; Enumeration type definitions
(enumeration_type_definition
  (enumeration_literal) @name.definition.constant) @definition.constant

; Variable declarations
(object_declaration
  name: (identifier) @name.definition.variable) @definition.variable

; Constant declarations
(object_declaration
  name: (identifier) @name.definition.constant) @definition.constant

; Procedure/Function calls
(procedure_call
  name: (identifier) @name.reference.call) @reference.call

(function_call
  name: (identifier) @name.reference.call) @reference.call

; Qualified name calls (Package.Procedure)
(selected_component
  selector: (identifier) @name.reference.call) @reference.call

; Package references
(selected_component
  prefix: (identifier) @name.reference.module) @reference.module

; Type references
(subtype_indication
  type: (identifier) @name.reference.type) @reference.type

; Generic package instantiation
(generic_instantiation
  name: (identifier) @name.definition.module) @definition.module

; Exception definitions
(exception_declaration
  name: (identifier) @name.definition.exception) @definition.exception

; Exception references
(raise_statement
  exception: (identifier) @name.reference.exception) @reference.exception

; Task definitions
(task_declaration
  name: (identifier) @name.definition.class) @definition.class

(task_body
  name: (identifier) @name.definition.class) @definition.class

; Protected type definitions
(protected_declaration
  name: (identifier) @name.definition.class) @definition.class

(protected_body
  name: (identifier) @name.definition.class) @definition.class

; Entry definitions (for tasks/protected types)
(entry_declaration
  name: (identifier) @name.definition.method) @definition.method
