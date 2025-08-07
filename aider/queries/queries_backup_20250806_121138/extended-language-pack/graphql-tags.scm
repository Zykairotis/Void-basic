; GraphQL query language tree-sitter tags query file

; Schema definition
(schema_definition) @definition.schema

; Type definitions
(object_type_definition
  name: (name) @name.definition.class) @definition.class

(interface_type_definition
  name: (name) @name.definition.interface) @definition.interface

(union_type_definition
  name: (name) @name.definition.union) @definition.union

(enum_type_definition
  name: (name) @name.definition.enum) @definition.enum

(scalar_type_definition
  name: (name) @name.definition.type) @definition.type

(input_object_type_definition
  name: (name) @name.definition.input) @definition.input

; Field definitions in types
(field_definition
  name: (name) @name.definition.field) @definition.field

; Input field definitions
(input_fields_definition
  (input_value_definition
    name: (name) @name.definition.field)) @definition.field

; Enum value definitions
(enum_value_definition
  name: (enum_value) @name.definition.constant) @definition.constant

; Arguments in field definitions
(arguments_definition
  (input_value_definition
    name: (name) @name.definition.argument)) @definition.argument

; Operation definitions
(operation_definition
  type: "query"
  name: (name) @name.definition.function) @definition.function

(operation_definition
  type: "mutation"
  name: (name) @name.definition.function) @definition.function

(operation_definition
  type: "subscription"
  name: (name) @name.definition.function) @definition.function

; Anonymous operations
(operation_definition
  type: "query") @definition.function

(operation_definition
  type: "mutation") @definition.function

(operation_definition
  type: "subscription") @definition.function

; Fragment definitions
(fragment_definition
  name: (fragment_name) @name.definition.fragment) @definition.fragment

; Variable definitions in operations
(variable_definition
  variable: (variable
    name: (name) @name.definition.variable)) @definition.variable

; Directive definitions
(directive_definition
  name: (name) @name.definition.directive) @definition.directive

; Type references
(named_type
  name: (name) @name.reference.type) @reference.type

; Field selections in operations
(field
  name: (name) @name.reference.field) @reference.field

; Field aliases
(field
  alias: (name) @name.definition.alias
  name: (name) @name.reference.field) @definition.alias

; Fragment spreads
(fragment_spread
  fragment_name: (fragment_name) @name.reference.fragment) @reference.fragment

; Inline fragment type conditions
(inline_fragment
  type_condition: (type_condition
    named_type: (named_type
      name: (name) @name.reference.type))) @reference.type

; Variable references
(variable
  name: (name) @name.reference.variable) @reference.variable

; Argument names in field calls
(argument
  name: (name) @name.reference.argument) @reference.argument

; Directive usages
(directive
  name: (name) @name.reference.directive) @reference.directive

; Union member types
(union_member_types
  (named_type
    name: (name) @name.reference.type)) @reference.type

; Interface implementations
(implements_interfaces
  (named_type
    name: (name) @name.reference.interface)) @reference.interface

; Type extensions
(object_type_extension
  name: (name) @name.reference.class) @reference.class

(interface_type_extension
  name: (name) @name.reference.interface) @reference.interface

(union_type_extension
  name: (name) @name.reference.union) @reference.union

(enum_type_extension
  name: (name) @name.reference.enum) @reference.enum

(scalar_type_extension
  name: (name) @name.reference.type) @reference.type

(input_object_type_extension
  name: (name) @name.reference.input) @reference.input

(schema_extension) @reference.schema

; Built-in scalar types
(named_type
  name: (name) @name.reference.builtin_type
  (#any-of? @name.reference.builtin_type "String" "Int" "Float" "Boolean" "ID")) @reference.builtin_type

; Built-in directives
(directive
  name: (name) @name.reference.builtin_directive
  (#any-of? @name.reference.builtin_directive "include" "skip" "deprecated" "specifiedBy")) @reference.builtin_directive

; List type references
(list_type
  (named_type
    name: (name) @name.reference.type)) @reference.type

; Non-null type references
(non_null_type
  (named_type
    name: (name) @name.reference.type)) @reference.type

(non_null_type
  (list_type
    (named_type
      name: (name) @name.reference.type))) @reference.type

; String values (for default values, descriptions, etc.)
(string_value) @reference.string

; Integer values
(int_value) @reference.number

; Float values
(float_value) @reference.number

; Boolean values
(boolean_value) @reference.boolean

; Null values
(null_value) @reference.null

; Enum values in usage
(enum_value) @name.reference.constant @reference.constant

; Object field values
(object_value
  (object_field
    name: (name) @name.reference.field)) @reference.field

; List values
(list_value) @reference.array

; Description strings
(description
  (string_value) @reference.description) @reference.description

; Schema root operation types
(root_operation_type_definition
  operation_type: "query"
  named_type: (named_type
    name: (name) @name.reference.query_type)) @reference.type

(root_operation_type_definition
  operation_type: "mutation"
  named_type: (named_type
    name: (name) @name.reference.mutation_type)) @reference.type

(root_operation_type_definition
  operation_type: "subscription"
  named_type: (named_type
    name: (name) @name.reference.subscription_type)) @reference.type

; Repeatable directive modifier
(directive_definition
  "repeatable") @reference.repeatable

; Field selection sets
(selection_set) @reference.selection_set

; Comments
(comment) @reference.comment
