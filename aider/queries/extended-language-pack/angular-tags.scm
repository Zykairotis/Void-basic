; Angular template tree-sitter tags query file

; Component selectors (custom elements)
(element
  (start_tag
    name: (tag_name) @name.reference.component
    (#match? @name.reference.component "^[a-z]+(-[a-z]+)+$")) @reference.component

; Structural directives (*ngFor, *ngIf, etc.)
(attribute
  name: (attribute_name) @name.reference.directive
  (#match? @name.reference.directive "^\\*ng[A-Z]") @reference.directive

; Regular directives (ngClass, ngStyle, etc.)
(attribute
  name: (attribute_name) @name.reference.directive
  (#match? @name.reference.directive "^ng[A-Z]") @reference.directive

; Property bindings [property]="value"
(attribute
  name: (attribute_name) @name.reference.property
  (#match? @name.reference.property "^\\[.*\\]$") @reference.property

; Event bindings (event)="handler"
(attribute
  name: (attribute_name) @name.reference.event
  (#match? @name.reference.event "^\\(.*\\)$") @reference.event

; Two-way bindings [(ngModel)]="value"
(attribute
  name: (attribute_name) @name.reference.binding
  (#match? @name.reference.binding "^\\[\\(.*\\)\\]$") @reference.binding

; Template reference variables #ref
(attribute
  name: (attribute_name) @name.definition.variable
  (#match? @name.definition.variable "^#") @definition.variable

; Angular pipes in interpolation
(text
  (#match? @text "\\{\\{.*\\|.*\\}\\}") @reference.pipe

; Interpolation expressions
(text
  (#match? @text "\\{\\{.*\\}\\}") @reference.expression

; Custom attribute directives
(attribute
  name: (attribute_name) @name.reference.directive
  (#match? @name.reference.directive "^app[A-Z]") @reference.directive

; Angular Material components
(element
  (start_tag
    name: (tag_name) @name.reference.component
    (#match? @name.reference.component "^mat-") @reference.component

; Angular CDK components
(element
  (start_tag
    name: (tag_name) @name.reference.component
    (#match? @name.reference.component "^cdk-") @reference.component

; Form directives
(attribute
  name: (attribute_name) @name.reference.directive
  (#any-of? @name.reference.directive "formGroup" "formControl" "formControlName" "formArray") @reference.directive

; Router directives
(attribute
  name: (attribute_name) @name.reference.directive
  (#any-of? @name.reference.directive "routerLink" "routerLinkActive" "routerOutlet") @reference.directive

; i18n attributes
(attribute
  name: (attribute_name) @name.reference.i18n
  (#match? @name.reference.i18n "^i18n") @reference.i18n

; Data binding attributes
(attribute
  name: (attribute_name) @name.reference.binding
  (#any-of? @name.reference.binding "ngModel" "ngValue" "ngSelected" "ngChecked" "ngDisabled") @reference.binding

; Animation triggers
(attribute
  name: (attribute_name) @name.reference.animation
  (#match? @name.reference.animation "^@") @reference.animation

; Template outlet contexts
(element
  (start_tag
    name: (tag_name) @name.reference.template
    (#eq? @name.reference.template "ng-template") @reference.template

; Container elements
(element
  (start_tag
    name: (tag_name) @name.reference.container
    (#eq? @name.reference.container "ng-container") @reference.container

; Content projection
(element
  (start_tag
    name: (tag_name) @name.reference.content
    (#eq? @name.reference.content "ng-content") @reference.content
