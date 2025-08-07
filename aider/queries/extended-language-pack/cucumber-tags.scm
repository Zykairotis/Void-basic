; Cucumber language tree-sitter tags query file
; Language type: testing
; Extensions: .feature

; Test scenarios
(scenario_definition
+  name: (identifier) @name.definition.scenario) @definition.scenario

; Test steps
(step_definition
+  text: (string) @name.definition.step) @definition.step

; Assertions
(assertion
+  condition: (identifier) @name.reference.assertion) @reference.assertion

; Test fixtures
(fixture_definition
+  name: (identifier) @name.definition.fixture) @definition.fixture

