; Astro language tree-sitter tags query file

; Component imports in frontmatter
(import_statement
  source: (string
    (string_fragment) @name.reference.module)
  (#match? @name.reference.module "\\.(astro|vue|svelte|jsx?|tsx?)$")) @reference.module

; Named imports (components)
(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name.reference.component
        (#match? @name.reference.component "^[A-Z]"))))) @reference.component

; Default imports (components)
(import_statement
  (import_clause
    (identifier) @name.reference.component
    (#match? @name.reference.component "^[A-Z]"))) @reference.component

; Function definitions in frontmatter
(function_declaration
  name: (identifier) @name.definition.function) @definition.function

; Arrow function assignments
(variable_declaration
  (variable_declarator
    name: (identifier) @name.definition.function
    value: (arrow_function))) @definition.function

; Variable declarations in frontmatter
(variable_declaration
  (variable_declarator
    name: (identifier) @name.definition.variable)) @definition.variable

; Const declarations
(lexical_declaration
  (variable_declarator
    name: (identifier) @name.definition.constant)) @definition.constant

; Interface definitions (TypeScript)
(interface_declaration
  name: (type_identifier) @name.definition.interface) @definition.interface

; Type alias definitions (TypeScript)
(type_alias_declaration
  name: (type_identifier) @name.definition.type) @definition.type

; Class definitions
(class_declaration
  name: (identifier) @name.definition.class) @definition.class

; Method definitions in classes
(method_definition
  name: (property_identifier) @name.definition.method) @definition.method

; Astro component usage in template
(jsx_element
  open_tag: (jsx_opening_element
    name: (identifier) @name.reference.component
    (#match? @name.reference.component "^[A-Z]"))) @reference.component

; Self-closing Astro components
(jsx_self_closing_element
  name: (identifier) @name.reference.component
  (#match? @name.reference.component "^[A-Z]")) @reference.component

; HTML elements
(jsx_element
  open_tag: (jsx_opening_element
    name: (identifier) @name.reference.element
    (#match? @name.reference.element "^[a-z]"))) @reference.element

; Astro directives (client:load, client:idle, etc.)
(jsx_attribute
  name: (property_identifier) @name.reference.directive
  (#match? @name.reference.directive "^client:")) @reference.directive

; Set directives
(jsx_attribute
  name: (property_identifier) @name.reference.directive
  (#match? @name.reference.directive "^set:")) @reference.directive

; Astro expressions in template
(jsx_expression
  (identifier) @name.reference.variable) @reference.variable

; Function calls in expressions
(jsx_expression
  (call_expression
    function: (identifier) @name.reference.call)) @reference.call

; Member expressions (Astro.props, etc.)
(jsx_expression
  (member_expression
    object: (identifier) @astro_global
    property: (property_identifier) @name.reference.property
    (#eq? @astro_global "Astro"))) @reference.property

; Props destructuring
(variable_declaration
  (variable_declarator
    name: (object_pattern
      (shorthand_property_identifier_pattern
        (identifier) @name.definition.variable)))) @definition.variable

; Props with types (TypeScript)
(variable_declaration
  (variable_declarator
    name: (object_pattern
      (shorthand_property_identifier_pattern
        (identifier) @name.definition.variable))
    type: (type_annotation))) @definition.variable

; Slot names
(jsx_element
  open_tag: (jsx_opening_element
    name: (identifier) @slot_element
    (jsx_attribute
      name: (property_identifier) @slot_attr
      value: (string
        (string_fragment) @name.definition.slot))
    (#eq? @slot_element "slot")
    (#eq? @slot_attr "name"))) @definition.slot

; Named slot usage
(jsx_attribute
  name: (property_identifier) @slot_attr
  value: (string
    (string_fragment) @name.reference.slot)
  (#eq? @slot_attr "slot")) @reference.slot

; CSS custom properties in style tag
(style_element
  (raw_text) @css_content
  (#contains? @css_content "--")) @definition.css_variable

; Event handlers
(jsx_attribute
  name: (property_identifier) @name.reference.event
  (#match? @name.reference.event "^on[A-Z]")) @reference.event

; Astro.glob patterns
(call_expression
  function: (member_expression
    object: (identifier) @astro_obj
    property: (property_identifier) @glob_method)
  arguments: (arguments
    (string
      (string_fragment) @name.reference.glob_pattern))
  (#eq? @astro_obj "Astro")
  (#eq? @glob_method "glob")) @reference.glob

; Define:vars (CSS variables passed to style)
(jsx_attribute
  name: (property_identifier) @define_vars
  (#eq? @define_vars "define:vars")) @reference.css_vars

; Fragment usage
(jsx_fragment) @reference.fragment

; Import.meta references
(member_expression
  object: (meta_property
    meta: (identifier) @import_meta
    property: (property_identifier) @meta_prop)
  property: (property_identifier) @name.reference.meta
  (#eq? @import_meta "import")
  (#eq? @meta_prop "meta")) @reference.meta

; Component props interface
(export_statement
  declaration: (interface_declaration
    name: (type_identifier) @name.definition.props
    (#eq? @name.definition.props "Props"))) @definition.props
