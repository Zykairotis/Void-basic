; JSON data language tree-sitter tags query file

; Object key definitions
(pair
  key: (string
    (string_content) @name.definition.key)) @definition.key

; Object key references in nested contexts
(pair
  key: (string
    (string_content) @name.reference.key)) @reference.key

; String values
(pair
  value: (string
    (string_content) @name.reference.string)) @reference.string

; Number values
(pair
  value: (number) @name.reference.number) @reference.number

; Boolean values
(pair
  value: (true) @name.reference.boolean) @reference.boolean

(pair
  value: (false) @name.reference.boolean) @reference.boolean

; Null values
(pair
  value: (null) @name.reference.null) @reference.null

; Array items
(array
  (string
    (string_content) @name.reference.array_item)) @reference.array_item

(array
  (number) @name.reference.array_item) @reference.array_item

(array
  (true) @name.reference.array_item) @reference.array_item

(array
  (false) @name.reference.array_item) @reference.array_item

(array
  (null) @name.reference.array_item) @reference.array_item

; Nested objects in arrays
(array
  (object) @name.reference.nested_object) @reference.nested_object

; Nested arrays
(array
  (array) @name.reference.nested_array) @reference.nested_array

; Common configuration keys
(pair
  key: (string
    (string_content) @name.definition.config_key
    (#any-of? @name.definition.config_key
      "name" "version" "description" "author" "license" "main" "entry"
      "scripts" "dependencies" "devDependencies" "peerDependencies"
      "config" "settings" "options" "parameters" "properties"
      "type" "format" "encoding" "charset" "lang" "language"
      "title" "subtitle" "keywords" "tags" "categories"
      "url" "homepage" "repository" "bugs" "issues"
      "host" "port" "path" "endpoint" "api" "service"
      "database" "db" "collection" "table" "schema"
      "user" "username" "email" "password" "token" "key" "secret"
      "timeout" "retry" "limit" "max" "min" "default"))) @definition.config_key

; File path references
(pair
  key: (string
    (string_content) @path_key
    (#any-of? @path_key "file" "path" "directory" "folder" "src" "dest"
      "input" "output" "source" "target" "location" "dir"
      "filename" "pathname" "filepath" "basePath" "rootPath"))
  value: (string
    (string_content) @name.reference.file_path)) @reference.file_path

; URL references
(pair
  value: (string
    (string_content) @name.reference.url
    (#match? @name.reference.url "^https?://|^ftp://|^ssh://|^file://"))) @reference.url

; Email addresses
(pair
  value: (string
    (string_content) @name.reference.email
    (#match? @name.reference.email "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"))) @reference.email

; Version strings (semantic versioning)
(pair
  key: (string
    (string_content) @version_key
    (#any-of? @version_key "version" "ver" "release" "tag"))
  value: (string
    (string_content) @name.reference.version
    (#match? @name.reference.version "^v?[0-9]+\\.[0-9]+\\.[0-9]+"))) @reference.version

; Date/time strings (ISO format)
(pair
  value: (string
    (string_content) @name.reference.datetime
    (#match? @name.reference.datetime "^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9]{2}:[0-9]{2}:[0-9]{2})?"))) @reference.datetime

; UUID strings
(pair
  value: (string
    (string_content) @name.reference.uuid
    (#match? @name.reference.uuid "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"))) @reference.uuid

; Color values (hex, rgb, etc.)
(pair
  value: (string
    (string_content) @name.reference.color
    (#match? @name.reference.color "^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$|^rgb\\(|^rgba\\(|^hsl\\(|^hsla\\("))) @reference.color

; Environment variable references
(pair
  value: (string
    (string_content) @name.reference.env_var
    (#match? @name.reference.env_var "\\$\\{[^}]+\\}|\\$[A-Z_][A-Z0-9_]*"))) @reference.env_var

; Base64 encoded data
(pair
  value: (string
    (string_content) @name.reference.base64
    (#match? @name.reference.base64 "^[A-Za-z0-9+/]*={0,2}$")
    (#not-match? @name.reference.base64 "^.{0,10}$"))) @reference.base64

; JSON Pointer references
(pair
  value: (string
    (string_content) @name.reference.json_pointer
    (#match? @name.reference.json_pointer "^/"))) @reference.json_pointer

; Glob patterns
(pair
  value: (string
    (string_content) @name.reference.glob
    (#match? @name.reference.glob "[*?\\[\\]]"))) @reference.glob

; Regular expressions
(pair
  key: (string
    (string_content) @regex_key
    (#any-of? @regex_key "pattern" "regex" "regexp" "match"))
  value: (string
    (string_content) @name.reference.regex)) @reference.regex

; MIME types
(pair
  key: (string
    (string_content) @mime_key
    (#any-of? @mime_key "contentType" "content-type" "mimeType" "mime"))
  value: (string
    (string_content) @name.reference.mime_type
    (#match? @name.reference.mime_type "^[a-z]+/[a-z0-9]+"))) @reference.mime_type

; Language codes
(pair
  key: (string
    (string_content) @lang_key
    (#any-of? @lang_key "lang" "language" "locale" "i18n" "l10n"))
  value: (string
    (string_content) @name.reference.language_code
    (#match? @name.reference.language_code "^[a-z]{2}(-[A-Z]{2})?$"))) @reference.language_code

; Package names (npm, etc.)
(pair
  key: (string
    (string_content) @name.definition.package_name
    (#not-match? @name.definition.package_name "^[A-Z]"))) @definition.package_name

; Schema references
(pair
  key: (string
    (string_content) @schema_key
    (#any-of? @schema_key "$schema" "schema" "$ref" "ref"))
  value: (string
    (string_content) @name.reference.schema)) @reference.schema

; Command line arguments/flags
(pair
  key: (string
    (string_content) @name.definition.cli_flag
    (#match? @name.definition.cli_flag "^-"))) @definition.cli_flag

; Error codes/status codes
(pair
  key: (string
    (string_content) @error_key
    (#any-of? @error_key "code" "status" "error" "errno"))
  value: (number) @name.reference.error_code) @reference.error_code

; Priority/weight/order values
(pair
  key: (string
    (string_content) @priority_key
    (#any-of? @priority_key "priority" "weight" "order" "index" "rank"))
  value: (number) @name.reference.priority) @reference.priority

; Boolean flags/switches
(pair
  key: (string
    (string_content) @name.definition.flag
    (#match? @name.definition.flag "^(is|has|can|should|enable|disable|allow|deny)[A-Z]|^[a-z]+ed$"))) @definition.flag
