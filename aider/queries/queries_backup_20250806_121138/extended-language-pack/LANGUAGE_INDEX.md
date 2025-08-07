# Extended Language Pack Index

This directory contains tree-sitter query files for 44 additional programming languages.

## Binary Languages

- **Wasm** (`.scm` file: `wasm-tags.scm`)
  - Extensions: .wasm, .wat
  - Features: functions, imports, exports, memory

## Blockchain Languages

- **Cairo** (`.scm` file: `cairo-tags.scm`)
  - Extensions: .cairo
  - Features: functions, structs, contracts

- **Move** (`.scm` file: `move-tags.scm`)
  - Extensions: .move
  - Features: modules, functions, structs, resources

- **Vyper** (`.scm` file: `vyper-tags.scm`)
  - Extensions: .vy
  - Features: contracts, functions, events, structs

## Build Languages

- **Bazel** (`.scm` file: `bazel-tags.scm`)
  - Extensions: BUILD, WORKSPACE, .bzl
  - Features: rules, targets, functions, macros

- **Cmake** (`.scm` file: `cmake-tags.scm`)
  - Extensions: CMakeLists.txt, .cmake
  - Features: functions, variables, targets, commands

- **Make** (`.scm` file: `make-tags.scm`)
  - Extensions: Makefile, .mk
  - Features: targets, variables, rules, dependencies

## Config Languages

- **Apache** (`.scm` file: `apache-tags.scm`)
  - Extensions: .conf, .htaccess
  - Features: directives, sections, variables

- **Crontab** (`.scm` file: `crontab-tags.scm`)
  - Extensions: .cron
  - Features: entries, commands, schedules

- **Nginx** (`.scm` file: `nginx-tags.scm`)
  - Extensions: .conf
  - Features: directives, blocks, variables

- **Systemd** (`.scm` file: `systemd-tags.scm`)
  - Extensions: .service, .socket, .timer
  - Features: sections, keys, values

## Data Languages

- **Bibtex** (`.scm` file: `bibtex-tags.scm`)
  - Extensions: .bib
  - Features: entries, fields, keys

## Database Languages

- **Cassandra** (`.scm` file: `cassandra-tags.scm`)
  - Extensions: .cql
  - Features: keyspaces, tables, types, functions

- **Cypher** (`.scm` file: `cypher-tags.scm`)
  - Extensions: .cypher, .cql
  - Features: nodes, relationships, patterns, functions

- **Mongodb** (`.scm` file: `mongodb-tags.scm`)
  - Extensions: .js
  - Features: collections, queries, aggregations, indexes

## Functional Languages

- **Erlang** (`.scm` file: `erlang-tags.scm`)
  - Extensions: .erl, .hrl
  - Features: modules, functions, records, macros

- **Fsharp** (`.scm` file: `fsharp-tags.scm`)
  - Extensions: .fs, .fsx, .fsi
  - Features: modules, functions, types, classes

## Game Languages

- **Angelscript** (`.scm` file: `angelscript-tags.scm`)
  - Extensions: .as
  - Features: classes, functions, interfaces, enums

- **Papyrus** (`.scm` file: `papyrus-tags.scm`)
  - Extensions: .psc
  - Features: scripts, functions, properties, events

- **Unrealscript** (`.scm` file: `unrealscript-tags.scm`)
  - Extensions: .uc
  - Features: classes, functions, states, events

## Hardware Languages

- **Systemverilog** (`.scm` file: `systemverilog-tags.scm`)
  - Extensions: .sv, .svh
  - Features: modules, classes, interfaces, packages

- **Verilog** (`.scm` file: `verilog-tags.scm`)
  - Extensions: .v, .vh
  - Features: modules, wires, registers, always_blocks

- **Vhdl** (`.scm` file: `vhdl-tags.scm`)
  - Extensions: .vhd, .vhdl
  - Features: entities, architectures, processes, signals

## Markup Languages

- **Creole** (`.scm` file: `creole-tags.scm`)
  - Extensions: .creole
  - Features: headings, links, lists, markup

- **Latex** (`.scm` file: `latex-tags.scm`)
  - Extensions: .tex, .sty, .cls
  - Features: commands, environments, packages, labels

- **Mediawiki** (`.scm` file: `mediawiki-tags.scm`)
  - Extensions: .wiki, .mediawiki
  - Features: templates, links, categories, magic_words

- **Textile** (`.scm` file: `textile-tags.scm`)
  - Extensions: .textile
  - Features: blocks, spans, links, lists

## Protocol Languages

- **Capnp** (`.scm` file: `capnp-tags.scm`)
  - Extensions: .capnp
  - Features: structs, interfaces, enums, constants

- **Flatbuffers** (`.scm` file: `flatbuffers-tags.scm`)
  - Extensions: .fbs
  - Features: tables, structs, enums, unions

## Query Languages

- **Xpath** (`.scm` file: `xpath-tags.scm`)
  - Extensions: .xpath
  - Features: expressions, functions, axes, predicates

- **Xquery** (`.scm` file: `xquery-tags.scm`)
  - Extensions: .xq, .xquery
  - Features: expressions, functions, modules, types

## Scientific Languages

- **Mathematica** (`.scm` file: `mathematica-tags.scm`)
  - Extensions: .nb, .m, .wl
  - Features: functions, variables, patterns, modules

- **Maxima** (`.scm` file: `maxima-tags.scm`)
  - Extensions: .mac, .wxm
  - Features: functions, variables, expressions

- **Octave** (`.scm` file: `octave-tags.scm`)
  - Extensions: .m
  - Features: functions, variables, scripts, classes

- **Sage** (`.scm` file: `sage-tags.scm`)
  - Extensions: .sage
  - Features: functions, classes, variables

- **Wolfram** (`.scm` file: `wolfram-tags.scm`)
  - Extensions: .wl, .m
  - Features: functions, patterns, modules, symbols

## Shell Languages

- **Elvish** (`.scm` file: `elvish-tags.scm`)
  - Extensions: .elv
  - Features: functions, commands, variables

- **Nushell** (`.scm` file: `nushell-tags.scm`)
  - Extensions: .nu
  - Features: commands, pipelines, functions, variables

- **Xonsh** (`.scm` file: `xonsh-tags.scm`)
  - Extensions: .xsh
  - Features: functions, commands, variables, aliases

## Specification Languages

- **Alloy** (`.scm` file: `alloy-tags.scm`)
  - Extensions: .als
  - Features: signatures, predicates, functions, facts

- **Tla** (`.scm` file: `tla-tags.scm`)
  - Extensions: .tla
  - Features: modules, operators, variables, actions

## Testing Languages

- **Cucumber** (`.scm` file: `cucumber-tags.scm`)
  - Extensions: .feature
  - Features: scenarios, steps, backgrounds, examples

## Theorem_Prover Languages

- **Coq** (`.scm` file: `coq-tags.scm`)
  - Extensions: .v
  - Features: definitions, theorems, inductives, modules

- **Lean4** (`.scm` file: `lean4-tags.scm`)
  - Extensions: .lean
  - Features: definitions, theorems, structures, inductive

