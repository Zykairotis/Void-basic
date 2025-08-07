; Crystal systems programming language tree-sitter tags query file

; Class definitions
(class_definition
  name: (constant) @name.definition.class) @definition.class

; Module definitions
(module_definition
  name: (constant) @name.definition.module) @definition.module

; Struct definitions
(struct_definition
  name: (constant) @name.definition.class) @definition.class

; Enum definitions
(enum_definition
  name: (constant) @name.definition.enum) @definition.enum

; Method definitions
(method_definition
  name: (identifier) @name.definition.method) @definition.method

; Function definitions (top-level methods)
(method_definition
  name: (identifier) @name.definition.function) @definition.function

; Getter method definitions
(getter_definition
  name: (identifier) @name.definition.property) @definition.property

; Setter method definitions
(setter_definition
  name: (identifier) @name.definition.property) @definition.property

; Property definitions
(property_definition
  name: (identifier) @name.definition.property) @definition.property

; Macro definitions
(macro_definition
  name: (identifier) @name.definition.macro) @definition.macro

; Alias definitions
(alias_definition
  name: (constant) @name.definition.type) @definition.type

; Constant definitions
(assignment
  left: (constant) @name.definition.constant) @definition.constant

; Variable assignments
(assignment
  left: (identifier) @name.definition.variable) @definition.variable

; Instance variable definitions
(assignment
  left: (instance_variable) @name.definition.variable) @definition.variable

; Class variable definitions
(assignment
  left: (class_variable) @name.definition.variable) @definition.variable

; Global variable definitions
(assignment
  left: (global_variable) @name.definition.variable) @definition.variable

; Parameter definitions
(parameter
  name: (identifier) @name.definition.parameter) @definition.parameter

; Block parameter definitions
(block_parameter
  name: (identifier) @name.definition.parameter) @definition.parameter

; Generic type parameter definitions
(type_parameter
  name: (constant) @name.definition.type_parameter) @definition.type_parameter

; Exception class definitions
(class_definition
  name: (constant) @name.definition.exception
  superclass: (constant) @superclass
  (#match? @superclass ".*Exception$")) @definition.exception

; Annotation definitions
(annotation_definition
  name: (constant) @name.definition.annotation) @definition.annotation

; Include statements
(include_statement
  (constant) @name.reference.module) @reference.module

; Extend statements
(extend_statement
  (constant) @name.reference.module) @reference.module

; Require statements
(require_statement
  (string_literal) @name.reference.require) @reference.require

; Method calls
(call_expression
  method: (identifier) @name.reference.call) @reference.call

; Method calls with receiver
(call_expression
  receiver: (identifier)
  method: (identifier) @name.reference.call) @reference.call

; Super calls
(super_expression) @reference.call

; Constant references
(constant) @name.reference.class @reference.class

; Type references
(type_annotation
  type: (constant) @name.reference.type) @reference.type

; Generic type instantiations
(generic_type
  type: (constant) @name.reference.type) @reference.type

; Union type references
(union_type
  (constant) @name.reference.type) @reference.type

; Nilable type references
(nilable_type
  (constant) @name.reference.type) @reference.type

; Variable references
(identifier) @name.reference.variable @reference.variable

; Instance variable references
(instance_variable) @name.reference.variable @reference.variable

; Class variable references
(class_variable) @name.reference.variable @reference.variable

; Global variable references
(global_variable) @name.reference.variable @reference.variable

; Symbol literals
(symbol_literal
  (identifier) @name.reference.symbol) @reference.symbol

; String interpolation
(string_interpolation
  (interpolation_expression
    (identifier) @name.reference.variable)) @reference.variable

; Hash literal keys
(hash_literal
  (hash_entry
    key: (symbol_literal
      (identifier) @name.definition.hash_key))) @definition.hash_key

; Named tuple keys
(named_tuple_literal
  (named_tuple_entry
    key: (identifier) @name.definition.tuple_key)) @definition.tuple_key

; Case statement when branches
(when_clause
  (identifier) @name.reference.variable) @reference.variable

; For loop variables
(for_statement
  variable: (identifier) @name.definition.variable) @definition.variable

; Block variables
(block_expression
  parameters: (block_parameters
    (identifier) @name.definition.variable)) @definition.variable

; Proc literal parameters
(proc_literal
  parameters: (parameter_list
    (parameter
      name: (identifier) @name.definition.parameter))) @definition.parameter

; Exception rescue clauses
(rescue_clause
  exception_variable: (identifier) @name.definition.variable) @definition.variable

(rescue_clause
  exception_type: (constant) @name.reference.exception) @reference.exception

; Ensure clauses
(ensure_clause) @reference.ensure

; Raise statements
(raise_expression
  (constant) @name.reference.exception) @reference.exception

; Type casting
(as_expression
  type: (constant) @name.reference.type) @reference.type

; Is-a type checks
(is_a_expression
  type: (constant) @name.reference.type) @reference.type

; Responds-to method checks
(responds_to_expression
  method: (identifier) @name.reference.call) @reference.call

; Typeof expressions
(typeof_expression
  (identifier) @name.reference.variable) @reference.variable

; Sizeof expressions
(sizeof_expression
  (constant) @name.reference.type) @reference.type

; Instance sizeof
(instance_sizeof_expression
  (constant) @name.reference.type) @reference.type

; Pointerof expressions
(pointerof_expression
  (identifier) @name.reference.variable) @reference.variable

; Annotation usages
(annotation
  name: (constant) @name.reference.annotation) @reference.annotation

; Macro calls
(macro_call
  name: (identifier) @name.reference.macro) @reference.macro

; Lib definitions (C bindings)
(lib_definition
  name: (constant) @name.definition.lib) @definition.lib

; Fun definitions (C function bindings)
(fun_definition
  name: (identifier) @name.definition.function) @definition.function

; Struct definitions in libs
(lib_definition
  (struct_definition
    name: (constant) @name.definition.struct)) @definition.struct

; Union definitions in libs
(lib_definition
  (union_definition
    name: (constant) @name.definition.union)) @definition.union

; Enum definitions in libs
(lib_definition
  (enum_definition
    name: (constant) @name.definition.enum)) @definition.enum

; Alias definitions in libs
(lib_definition
  (alias_definition
    name: (constant) @name.definition.type)) @definition.type

; C variable definitions
(lib_definition
  (variable_definition
    name: (identifier) @name.definition.variable)) @definition.variable

; Abstract class definitions
(abstract_class_definition
  name: (constant) @name.definition.class) @definition.class

; Abstract method definitions
(abstract_method_definition
  name: (identifier) @name.definition.method) @definition.method

; Private method definitions
(private_method_definition
  name: (identifier) @name.definition.method) @definition.method

; Protected method definitions
(protected_method_definition
  name: (identifier) @name.definition.method) @definition.method

; Visibility modifiers
(visibility_modifier) @reference.visibility

; Self references
(self_expression) @name.reference.self @reference.self

; Previous def references
(previous_def_expression) @reference.previous_def

; Uninitialized expressions
(uninitialized_expression
  (constant) @name.reference.type) @reference.type

; Splat operator references
(splat_expression
  (identifier) @name.reference.variable) @reference.variable

; Double splat operator references
(double_splat_expression
  (identifier) @name.reference.variable) @reference.variable

; Range expressions
(range_expression
  begin: (identifier) @name.reference.variable) @reference.variable

(range_expression
  end: (identifier) @name.reference.variable) @reference.variable

; Regex literal references
(regex_literal) @reference.regex

; Heredoc references
(heredoc_literal) @reference.heredoc
