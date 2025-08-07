; Apex language tree-sitter tags query file

; Class definitions
(class_declaration
  name: (identifier) @name.definition.class) @definition.class

; Interface definitions
(interface_declaration
  name: (identifier) @name.definition.interface) @definition.interface

; Enum definitions
(enum_declaration
  name: (identifier) @name.definition.enum) @definition.enum

; Enum constants
(enum_constant
  name: (identifier) @name.definition.constant) @definition.constant

; Trigger definitions
(trigger_declaration
  name: (identifier) @name.definition.function) @definition.function

; Method definitions
(method_declaration
  name: (identifier) @name.definition.method) @definition.method

; Constructor definitions
(constructor_declaration
  name: (identifier) @name.definition.method) @definition.method

; Property definitions
(property_declaration
  name: (identifier) @name.definition.property) @definition.property

; Field definitions
(field_declaration
  declarator: (variable_declarator
    name: (identifier) @name.definition.variable)) @definition.variable

; Static initializer blocks
(static_initializer) @definition.method

; Local variable declarations
(local_variable_declaration
  declarator: (variable_declarator
    name: (identifier) @name.definition.variable)) @definition.variable

; Parameter declarations
(formal_parameter
  name: (identifier) @name.definition.variable) @definition.variable

; Method invocations
(method_invocation
  name: (identifier) @name.reference.call) @reference.call

; Object creation
(object_creation_expression
  type: (type_identifier) @name.reference.class) @reference.class

; Type references
(type_identifier) @name.reference.type @reference.type

; Super class references
(superclass
  (type_identifier) @name.reference.class) @reference.class

; Interface implementations
(super_interfaces
  (interface_type_list
    (type_identifier) @name.reference.interface)) @reference.interface

; Field access
(field_access
  field: (identifier) @name.reference.property) @reference.property

; Annotation types
(annotation
  name: (identifier) @name.reference.annotation) @reference.annotation

; Exception types in catch clauses
(catch_clause
  parameter: (catch_formal_parameter
    type: (type_identifier) @name.reference.exception)) @reference.exception

; SOQL queries
(soql_query) @reference.query

; SOSL queries
(sosl_query) @reference.query

; SObject types in SOQL
(soql_query
  (from_clause
    (identifier) @name.reference.sobject)) @reference.sobject

; Custom settings and objects
(type_identifier
  (#match? @type_identifier ".*__c$")) @name.reference.custom_object

; Test method annotations
(annotation
  name: (identifier) @name.reference.test
  (#any-of? @name.reference.test "Test" "IsTest" "TestSetup")) @reference.test

; Sharing keywords
(sharing_modifier) @reference.sharing

; Database operations
(method_invocation
  object: (identifier) @database_class
  name: (identifier) @name.reference.database_method
  (#any-of? @database_class "Database" "System")
  (#any-of? @name.reference.database_method "insert" "update" "delete" "query" "queryWithBinds")) @reference.database

; Trigger context variables
(identifier
  (#any-of? @identifier "Trigger.new" "Trigger.old" "Trigger.newMap" "Trigger.oldMap"
                       "Trigger.isInsert" "Trigger.isUpdate" "Trigger.isDelete"
                       "Trigger.isBefore" "Trigger.isAfter" "Trigger.isUndelete")) @name.reference.trigger_context

; Apex collections
(object_creation_expression
  type: (generic_type
    (type_identifier) @collection_type
    (#any-of? @collection_type "List" "Set" "Map"))) @reference.collection

; Future method annotations
(annotation
  name: (identifier) @name.reference.future
  (#eq? @name.reference.future "future")) @reference.future

; AuraEnabled annotations
(annotation
  name: (identifier) @name.reference.aura
  (#eq? @name.reference.aura "AuraEnabled")) @reference.aura

; RemoteAction annotations
(annotation
  name: (identifier) @name.reference.remote
  (#eq? @name.reference.remote "RemoteAction")) @reference.remote

; InvocableMethod annotations
(annotation
  name: (identifier) @name.reference.invocable
  (#eq? @name.reference.invocable "InvocableMethod")) @reference.invocable

; HttpGet/HttpPost annotations for REST services
(annotation
  name: (identifier) @name.reference.http
  (#any-of? @name.reference.http "HttpGet" "HttpPost" "HttpPut" "HttpDelete" "HttpPatch")) @reference.http

; Webservice methods
(method_declaration
  (modifiers
    (modifier
      (#eq? @modifier "webservice")))
  name: (identifier) @name.definition.webservice) @definition.webservice

; Custom exception classes
(class_declaration
  name: (identifier) @name.definition.exception
  superclass: (superclass
    (type_identifier)
    (#match? @type_identifier ".*Exception$"))) @definition.exception
