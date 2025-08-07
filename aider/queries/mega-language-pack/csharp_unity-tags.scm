; Tree-sitter query file for Csharp_Unity
; Language: Csharp_Unity
; Type: Game
; Description: C# for Unity game engine
; Version: 1.0
; Generated: 2025-08-06
; Features: monobehaviours, scriptableobjects, coroutines


; Variable definitions
(variable_declaration
  name: (identifier) @name.definition.variable) @definition.variable

(assignment_expression
  left: (identifier) @name.definition.variable) @definition.variable

; Variable references
(identifier) @name.reference.variable @reference.variable

; Type definitions
(type_definition
  name: (identifier) @name.definition.type) @definition.type

(type_declaration
  name: (identifier) @name.definition.type) @definition.type

; Language-specific constructs

; Game object references
(game_object
  name: (identifier) @name.reference.gameobject) @reference.gameobject

; Component references
(component_reference
  name: (identifier) @name.reference.component) @reference.component

; Script references
(script_reference
  name: (identifier) @name.reference.script) @reference.script

; Comments
(comment) @comment

; Strings
(string) @string

; Numbers
(number) @number
(integer) @number
(float) @number
