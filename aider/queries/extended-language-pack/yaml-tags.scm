; YAML configuration language tree-sitter tags query file

; Block mapping keys (property definitions)
(block_mapping_pair
  key: (flow_node
    (plain_scalar) @name.definition.key) @definition.key

(block_mapping_pair
  key: (flow_node
    (single_quote_scalar) @name.definition.key) @definition.key

(block_mapping_pair
  key: (flow_node
    (double_quote_scalar) @name.definition.key) @definition.key

; Flow mapping keys (inline object keys)
(flow_mapping
  (flow_pair
    key: (flow_node
      (plain_scalar) @name.definition.key)) @definition.key

(flow_mapping
  (flow_pair
    key: (flow_node
      (single_quote_scalar) @name.definition.key)) @definition.key

(flow_mapping
  (flow_pair
    key: (flow_node
      (double_quote_scalar) @name.definition.key)) @definition.key

; Anchor definitions (&anchor)
(anchor_node
  (anchor_name) @name.definition.anchor) @definition.anchor

; Alias references (*alias)
(alias_node
  (alias_name) @name.reference.alias) @reference.alias

; YAML tags (!tag)
(tag
  (tag_name) @name.reference.tag) @reference.tag

; Document markers
(document_start) @definition.document
(document_end) @definition.document

; Comments
(comment) @reference.comment

; String values (for configuration analysis)
(block_mapping_pair
  value: (flow_node
    (plain_scalar) @name.reference.value) @reference.value

(block_mapping_pair
  value: (flow_node
    (single_quote_scalar) @name.reference.string) @reference.string

(block_mapping_pair
  value: (flow_node
    (double_quote_scalar) @name.reference.string) @reference.string

; Boolean values
(block_mapping_pair
  value: (flow_node
    (plain_scalar) @name.reference.boolean
    (#match? @name.reference.boolean "^(true|false|yes|no|on|off)$")) @reference.boolean

; Numeric values
(block_mapping_pair
  value: (flow_node
    (plain_scalar) @name.reference.number
    (#match? @name.reference.number "^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$")) @reference.number

; Null values
(block_mapping_pair
  value: (flow_node
    (plain_scalar) @name.reference.null
    (#match? @name.reference.null "^(null|~|)$")) @reference.null

; Environment variable references
(block_mapping_pair
  value: (flow_node
    (plain_scalar) @name.reference.env_var
    (#match? @name.reference.env_var "\\$\\{[^}]+\\}|\\$[A-Z_][A-Z0-9_]*") @reference.env_var

(block_mapping_pair
  value: (flow_node
    (double_quote_scalar) @name.reference.env_var
    (#match? @name.reference.env_var "\\$\\{[^}]+\\}|\\$[A-Z_][A-Z0-9_]*") @reference.env_var

; Block sequence items (list items)
(block_sequence_item
  (flow_node
    (plain_scalar) @name.reference.list_item) @reference.list_item

; Flow sequence items (inline list items)
(flow_sequence
  (flow_node
    (plain_scalar) @name.reference.list_item) @reference.list_item

; Multiline strings (literal and folded)
(block_scalar) @name.reference.multiline_string @reference.multiline_string

; Common configuration keys (for better semantic understanding)
(block_mapping_pair
  key: (flow_node
    (plain_scalar) @name.definition.config_key
    (#any-of? @name.definition.config_key)
      "name" "version" "description" "author" "license" "dependencies"
      "scripts" "main" "entry" "output" "build" "test" "dev" "prod"
      "environment" "env" "config" "settings" "options" "parameters"
      "host" "port" "url" "path" "database" "db" "user" "password"
      "token" "key" "secret" "api_key" "timeout" "retry" "limit")) @definition.config_key

; File path references
(block_mapping_pair
  key: (flow_node
    (plain_scalar) @path_key
    (#any-of? @path_key "file" "path" "directory" "folder" "output" "input")
  value: (flow_node
    (plain_scalar) @name.reference.file_path
    (#match? @name.reference.file_path "\\./|\\.\\./") @reference.file_path

(block_mapping_pair
  key: (flow_node
    (plain_scalar) @path_key
    (#any-of? @path_key "file" "path" "directory" "folder" "output" "input")
  value: (flow_node
    (single_quote_scalar) @name.reference.file_path) @reference.file_path

(block_mapping_pair
  key: (flow_node
    (plain_scalar) @path_key
    (#any-of? @path_key "file" "path" "directory" "folder" "output" "input")
  value: (flow_node
    (double_quote_scalar) @name.reference.file_path) @reference.file_path

; URL references
(block_mapping_pair
  value: (flow_node
    (plain_scalar) @name.reference.url
    (#match? @name.reference.url "^https?://|^ftp://|^ssh://") @reference.url

(block_mapping_pair
  value: (flow_node
    (single_quote_scalar) @name.reference.url
    (#match? @name.reference.url "^https?://|^ftp://|^ssh://") @reference.url

(block_mapping_pair
  value: (flow_node
    (double_quote_scalar) @name.reference.url
    (#match? @name.reference.url "^https?://|^ftp://|^ssh://") @reference.url

; Version strings
(block_mapping_pair
  key: (flow_node
    (plain_scalar) @version_key
    (#any-of? @version_key "version" "ver" "release")
  value: (flow_node
    (plain_scalar) @name.reference.version) @reference.version

; Include/import references
(block_mapping_pair
  key: (flow_node
    (plain_scalar) @include_key
    (#any-of? @include_key "include" "import" "extends" "inherit")
  value: (flow_node
    (plain_scalar) @name.reference.include) @reference.include

; Service/container names (for Docker Compose, Kubernetes, etc.)
(block_mapping_pair
  key: (flow_node
    (plain_scalar) @name.definition.service) @definition.service
