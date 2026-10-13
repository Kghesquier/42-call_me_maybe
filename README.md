*This project has been created as part of the 42 curriculum by kghesqui.*

# call me maybe

## Description

**call me maybe** turns natural-language requests into structured function
calls. Given *"What is the sum of 2 and 3?"* and a list of available
functions, the program does not answer `5`: it outputs

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

It uses a small language model (**Qwen/Qwen3-0.6B**) and **constrained
decoding**: at every generation step, the tokens that would break the JSON
structure or the function schema are masked out, so the output is always
valid JSON matching the definitions, while the model still decides *which*
function to call and *what* the argument values are.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make install        # = uv sync (installs numpy, pydantic and llm_sdk)
make run            # = uv run python -m src  (default paths)
make debug          # run under pdb
make lint           # flake8 + mypy with the required flags
make lint-strict    # flake8 + mypy --strict
make test           # unit tests (pytest)
make clean          # remove caches
```

The first run downloads the model from the Hugging Face Hub (~1.2 GB).

### Example usage

```bash
uv run python -m src
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
uv run python -m src --verbose      # print every generated token
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |
| `--model` | `Qwen/Qwen3-0.6B` |
| `--verbose` | off |

Progress and errors go to stderr; the output file only ever contains the
JSON array of results.

## Algorithm explanation

### 1. Reading the vocabulary

`get_path_to_vocab_file()` gives Qwen's `vocab.json` (token → id). Qwen
uses byte-level BPE: every byte is written as a printable character
(`Ġ` = space, `Ċ` = newline). `src/vocabulary.py` reverses that mapping to
know the **real text of every token id**. Tokens that are not complete UTF-8
sequences get an empty text and are never allowed.

### 2. Forced structure, free values

The JSON skeleton is never left to the model. The decoder builds the
context step by step; parts in quotes below are **forced** (encoded and
appended directly), the others are **generated under constraints**:

```
<prompt> '{"name": "' NAME '", "parameters": {"a": ' NUMBER ', "b": ' NUMBER '}}'
```

Forcing the skeleton is both safer (structure cannot be wrong) and faster
(no model call for those tokens).

### 3. One masking step

For every free token:

1. `get_logits_from_input_ids(context)` returns one logit per token;
2. the decoder computes the set of **allowed** token ids;
3. every other logit is set to `-inf`;
4. the token with the highest remaining logit is chosen (greedy).

### 4. What "allowed" means

| Value | Rule |
|---|---|
| function name | `generated + token` must be a prefix of one of `name + '"'`. The closing quote makes names prefix-free (`fn_add` vs `fn_add_numbers`). When a single name remains it is completed without calling the model. |
| `number` | token made only of `0-9 . - + e E`, and `generated + token` must match the prefix grammar `-?\d+(\.\d*)?([eE][+-]?\d{0,3})?`. Once the number is complete, tokens starting with `,`, `}` or whitespace are also allowed: picking one means "done". |
| `integer` | same with `-?\d+`. |
| `string` | a JSON-string state machine (normal / after `\` / inside `\uXXXX`). Tokens without `"`, `\` or control characters are always valid in the normal state; the others are run through the state machine. A token containing an unescaped `"` closes the string. Invalid escapes such as `\d` are impossible, so a regex is written `\\d+` and decodes to `\d+`. |
| `boolean` | choice between `true` and `false`. |

Values are then parsed with Python (`float`, `int`, `json.loads`) and
stored in a pydantic `FunctionCall`, which is serialised with
`allow_nan=False`.

## Design decisions

- **Greedy decoding** (argmax): deterministic and reproducible results.
- **Pre-classification of the vocabulary** (`plain`, `special`, `number`,
  `number end`, by first character) done once at start-up with numpy
  arrays, so each step only checks a few thousand tokens in Python instead
  of 151 000.
- **Closing quote kept out of the context**: when a token closes a string,
  only its part before the quote is kept; the quote is re-added with the
  next forced segment. This keeps the context identical to a clean JSON.
- **Qwen3 chat template with an empty `<think></think>` block**: disables
  the reasoning mode so the model answers directly.
- **Prompt**: short list of signatures (`fn(a: number)`) + descriptions,
  and a reminder that backslashes are doubled in JSON strings.
- **pydantic everywhere**: input files, parameters, results and the decoder
  itself are pydantic models; invalid inputs give precise messages.
- **Unknown argument types** are generated as strings, with a warning.
- **Safety limits**: 64 tokens per string, 32 characters per number;
  truncated values are repaired into valid ones.

## Performance analysis

- **Validity**: 100 % by construction. Every token is checked against the
  grammar, so the output always parses and always matches the schema. The
  tests run the decoder with *random* logits on many seeds to prove it.
- **Accuracy**: depends on the model's choices inside the constraints;
  the forced structure removes the most common failure of small models
  (invalid or chatty output). Prompts with an obvious function are
  reliably correct; ambiguous prompts are the main source of errors.
- **Speed**: one forward pass per *generated* token only (structure and
  forced name endings cost nothing). Decoder overhead is below one second
  per prompt; the whole example set takes well under the 5-minute limit on
  CPU. `--verbose` shows how many tokens were allowed at each step.

## Challenges faced

- **Token ≠ character**: a single token can contain several characters,
  part of a number and a comma, or a quote in the middle. Every check is
  done on `generated + token_text`, never on single characters.
- **Byte-level vocabulary**: `vocab.json` is not plain text; the GPT-2
  byte table had to be rebuilt to read it.
- **Escapes in strings**: regex arguments contain backslashes; a small
  state machine allows only legal JSON escapes.
- **Prefix names**: `fn_add` / `fn_add_numbers` — solved with the closing
  quote in the options.
- **Speed in pure Python**: solved by classifying tokens once with numpy.

## Testing strategy

`make test` runs pytest with a fake model (`tests/fake_llm.py`) that
exposes exactly the SDK methods and writes a byte-level `vocab.json`:

- unit tests of the vocabulary decoding, the string state machine and the
  repair functions;
- a "guided" fake model that wants to write a given JSON: the decoder must
  return exactly that call (numbers, negative integers, booleans, regex
  escapes, prefix function names, functions without arguments);
- a fuzz test with random logits on 25 seeds: the output must always
  match the schema and be serialisable;
- input handling: missing file, invalid JSON, wrong structure, invalid
  entries, and an end-to-end run of the program.

## Resources

- Hugging Face — [Tokenizers summary (BPE, byte-level BPE)](https://huggingface.co/docs/transformers/tokenizer_summary)
- OpenAI GPT-2 source — `bytes_to_unicode` in [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- Willard & Louf, [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702) (2023)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [pydantic documentation](https://docs.pydantic.dev/)

### Use of AI

An AI assistant (Claude) was used to draft the implementation, the test
suite and this README. All of the code was then reviewed, run and
explained line by line by the author before submission.
