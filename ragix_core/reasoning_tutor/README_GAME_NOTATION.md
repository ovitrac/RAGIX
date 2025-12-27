# LLM Reasoning Games: Notation & Meta-Cognitive Detection

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-12-23
**Status:** Milestone Document

---

## 1. A Turning Point: From Execution to Meta-Cognition

This document marks a fundamental shift in how we evaluate LLM reasoning capabilities.

### The Old Paradigm: Execution Loop
```
while not done:
    action = llm.generate()
    result = bash.execute(action)
    if result.returncode != 0:
        penalize()  # Watch bash return codes
```

### The New Paradigm: Meta-Cognitive System
```
while not done:
    action = llm.generate()
    result = execute(action)
    detector.record(action, result, pcg_state)

    failure = detector.detect()  # Watch GAME DYNAMICS
    if failure:
        diagnose(failure)
        intervene(failure)
```

**Key Insight:** We now detect *patterns of behavior*, not just command success/failure.

---

## 2. Failure Taxonomy: The Diagnosis Table

| Failure Type | Clinical Term | Diagnosis | Intervention |
|-------------|---------------|-----------|--------------|
| `REPETITION_LOOP` | **Perseveration** | Syntactic aphasia - repeating same command | Alternative syntax card |
| `CIRCULAR_PATTERN` | **Disorientation** | Strategic confusion - cycling without progress | Tactical guidance card |
| `PROGRESS_STALL` | **Confabulation** | Verbose but unproductive - reading without learning | Synthesis card |
| `EXPLICIT_ERROR` | **Agnosia** | Failure to recognize error patterns | Error recovery card |
| `EXHAUSTION` | **Resource Depletion** | All strategies tried, none worked | Capability extension |

### Clinical Analogy

These patterns mirror cognitive phenomena in human reasoning:

- **Perseveration** (REPETITION): Like a patient repeatedly trying the same failed action
- **Disorientation** (CIRCULAR): Like wandering without a mental map
- **Confabulation** (STALL): Producing plausible but meaningless output
- **Agnosia** (ERROR): Inability to recognize what went wrong

**Important:** These diagnoses say nothing about "intelligence" - they characterize *behavioral patterns* that indicate when assistance is needed.

---

## 3. Game Notation: Algebraic Notation for LLM Reasoning

Inspired by chess notation, we introduce a formal language to record and analyze LLM reasoning games.

### 3.1 Basic Notation Elements

#### Action Symbols
| Symbol | Action | Example |
|--------|--------|---------|
| `R` | Read file | `R:main.py` |
| `S` | Search content | `S:"EUREKA"` |
| `F` | Find file | `F:*.py` |
| `L` | List directory | `L:src/` |
| `C` | Count lines | `C:*.py` |
| `G` | Grep pattern | `G:import/core.py` |
| `E` | Echo/Answer | `E:"file_delta.txt"` |
| `X` | Execute raw | `X:git status` |
| `?` | Card menu access | `?3` (chose card #3) |

#### Result Markers
| Symbol | Meaning |
|--------|---------|
| `✓` | Success (command worked, evidence gained) |
| `✗` | Error (command failed) |
| `○` | Neutral (command worked but no progress) |
| `⊕` | Goal progress (moved closer to solution) |
| `◉` | Goal achieved |

#### Quality Annotations (like chess !!, !, ?!, ?, ??)
| Symbol | Meaning | Description |
|--------|---------|-------------|
| `!!` | Brilliant | Optimal action, shows insight |
| `!` | Good | Correct direction |
| `!?` | Interesting | Creative but risky |
| `?!` | Dubious | Questionable choice |
| `?` | Mistake | Suboptimal action |
| `??` | Blunder | Terrible choice |
| `⟳` | Repetition | Same action repeated |
| `↻` | Circular | Part of a cycle pattern |
| `∅` | Stall | No PCG progress |

#### State Indicators
| Symbol | Meaning |
|--------|---------|
| `[+N]` | Score change |
| `{n}` | PCG nodes count |
| `T:n` | Turn number |
| `⚠` | Failure detected |

### 3.2 Move Notation Format

```
T:n  Action:target  Result[Score]{PCG}  Annotation
```

**Examples:**
```
T:1  S:"EUREKA"      ✓[+100]{2}    !   # Good opening - direct search
T:2  R:data/log.txt  ✓[+100]{4}    !!  # Brilliant - found the needle
T:3  E:"log.txt"     ◉[+250]{5}        # Goal achieved

T:1  L:.             ✓[+100]{1}    ?   # Dubious - too generic
T:2  L:.             ✗[-30]{1}     ⟳?? # Blunder - repeated same action
T:3  L:data/         ✓[+100]{2}        # Recovery
T:4  L:.             ✗[-30]{2}     ↻   # Circular pattern emerging
```

### 3.3 Game Header (like PGN in chess)

```
[Event "Benchmark 01: Find Needle"]
[Date "2025.12.23"]
[White "granite3.1-moe:3b"]
[Black "Tutor (Interpreter)"]
[Result "1-0"]  # LLM won / 0-1 LLM lost / ½-½ partial
[OptimalTurns "3"]
[ActualTurns "5"]
[FinalScore "+180"]
[Failures "1:repetition_loop@T4"]

1. S:"EUREKA" ✓! {grep -r EUREKA .}
2. R:data/delta.txt ✓!! {cat data/file_delta.txt}
3. E:"file_delta.txt" ◉ 1-0
```

### 3.4 Compact Notation for Quick Comparison

For rapid comparison, use single-line format:
```
granite3.1@B01: S!→R!!→E◉ [3T +350] 1-0
mistral:7b@B01: L?→L⟳??→?3→R!→E◉ [5T +180] 1-0
```

---

## 4. Implementation: FailureDetector

The detection system is implemented in `failure_detector.py`:

```python
class FailureDetector:
    """
    Meta-cognitive detection of stuck states.

    Detection Priority:
    1. EXPLICIT_ERROR  - Command fails 3+ times consecutively
    2. REPETITION_LOOP - Same action repeated 3+ times
    3. CIRCULAR_PATTERN - A→B→A→B or A→B→C→A→B→C cycling
    4. PROGRESS_STALL  - No new PCG nodes in 4+ turns
    5. EXHAUSTION      - All cards tried without progress
    """
```

### Key Design: PCG-Based Progress Detection

```python
def _detect_progress_stall(self) -> bool:
    """
    No new PCG nodes added in N turns.

    CRITICAL: "Empty outputs" is too weak a signal.
    A model can "babble" (verbose ls, cat commands) without progress.

    STRICT: Compare len(pcg.nodes) at turn T vs T-N.
    No new Truth/Observation/Entity = no progress.
    """
    current_nodes = self.pcg_node_counts[-1]
    nodes_n_turns_ago = self.pcg_node_counts[-self.stall_threshold]
    return current_nodes == nodes_n_turns_ago
```

### Differentiated Interventions

```python
def get_fat_llm_instruction(failure: FailureContext, goal: str) -> str:
    """Generate appropriate Fat LLM prompt based on failure type."""

    if failure.failure_type == FailureType.REPETITION_LOOP:
        # DIAGNOSIS: Syntactic aphasia
        # ACTION: Propose alternative syntax
        return "CARD REQUEST: Alternative Syntax..."

    elif failure.failure_type == FailureType.CIRCULAR_PATTERN:
        # DIAGNOSIS: Strategic disorientation
        # ACTION: Propose tactical guidance
        return "CARD REQUEST: Strategic Direction..."

    elif failure.failure_type == FailureType.PROGRESS_STALL:
        # DIAGNOSIS: Confabulation (verbose but unproductive)
        # ACTION: Propose synthesis/consolidation
        return "CARD REQUEST: Synthesis/Consolidation..."
```

---

## 5. Comparative Game Records

### Legend
- **W**: White (LLM under test)
- **B**: Black (Tutor/Interpreter)
- **T**: Turn number
- **Score**: Cumulative score at end of turn

*Games recorded below using formal notation.*

---

## Match Day: 2025-12-23 — granite3.1-moe:3b vs mistral:7b-instruct

### Overall Standings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│        MODEL COMPARISON: granite3.1-moe:3b vs mistral:7b-instruct          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Benchmark      │ granite3.1-moe:3b     │ mistral:7b-instruct        │ Winner │
├────────────────┼───────────────────────┼────────────────────────────┼────────┤
│ B01 Needle     │ 0-1 [10T, +390]       │ 0-1 [10T, +330]            │ Draw   │
│ B04 Chain      │ 1-0 [6T, +180]        │ 0-1 [12T, -325]            │ granite│
│ B06 Memory     │ 0-1 [10T, +170]       │ 0-1 [10T, +535]            │ mistral│
├────────────────┼───────────────────────┼────────────────────────────┼────────┤
│ TOTAL          │ 1 Win, 2 Losses       │ 0 Wins, 3 Losses           │granite │
│ Aggregate Score│ +740                  │ +540                       │        │
│ Failures Det.  │ 2 (circ, err)         │ 7 (circ, rep, err×4, rep)  │        │
│ Own Solutions  │ 8                     │ 13                         │        │
│ Card Solutions │ 6                     │ 2                          │        │
│ Avg Latency    │ 1759 ms/turn          │ 5876 ms/turn               │granite │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The smaller model (3B) outperformed the larger model (7B):
- Faster (3x lower latency)
- Better at using cards when stuck
- Won the only winnable game (B04)

---

### Game 1: B01 Find Needle - granite3.1-moe:3b

```
[Event "B01: Find Needle in Haystack"]
[Date "2025.12.23"]
[White "granite3.1-moe:3b"]
[Black "Tutor"]
[Result "0-1"]
[OptimalTurns "3"]
[ActualTurns "10"]
[FinalScore "+390"]
[FailuresDetected "1:circular_pattern@T6"]
[Goal "Find file containing EUREKA_SECRET_42"]

1. G:"EUREKA"/data ○[+100]{1}     # grep -rl worked, no output shown
2. G:"EUREKA"/data ⟳○[+70]{2} ?   # Repetition - same command
3. X:grep ✗[-15]{3} ?             # Syntax error - incomplete
4. G:"EUREKA"/data ⟳○[+55]{4} ??  # Perseveration
5. G:"EUREKA"/data ⟳○[+25]{5} ??  # Still perseverating
6. X:grep ⟳✗[-45]{6} ⚠↻          # CIRCULAR_PATTERN detected!
7. X:"250 points" ✗[-15]{7}       # Confused by score in prompt
8. E:"file_alpha.txt" ○[+100]{8}  # Guessing (wrong file)
9. X:grep ⟳✗[-45]{9} ??           # Back to broken pattern
10. E:"YOUR_ANSWER" ○[+100]{10}   # Literal interpretation

Result: 0-1 (Failed - never found correct file)
Diagnosis: Perseveration (T2-T5), circular pattern (T6), confabulation (T8-T10)
Pattern: G→G→X→G→G→X→?→E→X→E (chaotic)
```

---

### Game 2: B01 Find Needle - mistral:7b-instruct

```
[Event "B01: Find Needle in Haystack"]
[Date "2025.12.23"]
[White "mistral:7b-instruct"]
[Black "Tutor"]
[Result "0-1"]
[OptimalTurns "3"]
[ActualTurns "10"]
[FinalScore "+330"]
[FailuresDetected "2:circular_pattern@T6,repetition_loop@T10"]
[Goal "Find file containing EUREKA_SECRET_42"]

1. X:grep ✗[-15]{1} ?             # Incomplete grep command
2. X:grep ⟳✗[-45]{2} ??           # Same error repeated
3. G:"EUREKA"/data ✓[+100]{3} !   # Finally correct syntax
4. X:grep ⟳✗[-45]{4} ??           # Regressed to broken command
5. X:grep ⟳✗[-45]{5} ??           # Perseveration
6. G:"EUREKA"/data ⟳○[+20]{6} ⚠↻ # CIRCULAR detected
7. G:"EUREKA"/data ⟳○[+90]{7}     # Still cycling
8. G:-r/data ✓[+100]{8}           # Variant tried
9. G:-r/data ⟳○[+60]{9}           # Repetition begins
10. G:-r/data ⟳○[+30]{10} ⚠⟳     # REPETITION_LOOP detected

Result: 0-1 (Failed)
Diagnosis: Early agnosia (T1-T2), recovery (T3), relapse (T4-T5), cycling (T6-10)
Pattern: X→X→G→X→X→G→G→G→G→G (two distinct error modes)
Latency: 5428 ms/turn (3.6x slower than granite)
```

---

### Game 3: B04 Verification Chain - granite3.1-moe:3b

```
[Event "B04: Verification Chain"]
[Date "2025.12.23"]
[White "granite3.1-moe:3b"]
[Black "Tutor"]
[Result "1-0"]
[OptimalTurns "6"]
[ActualTurns "6"]
[FinalScore "+180"]
[PathEfficiency "100%"]
[FailuresDetected "0"]
[Goal "Verify main.py imports resolve"]

1. X:main.py ✗[-15]{1} ?          # Tried to execute as command
2. X:main.py ⟳✗[-45]{2} ??        # Perseveration
3. ?1→R:project/main.py ✓[+40]{3}! # Card rescue - found imports!
4. X:main.py ⟳✗[-45]{4} ?!        # Relapse (but card activated)
5. X:main.py ⟳✗[-45]{5} ??        # Perseveration
6. ?1→R:project/core.py ✓⊕[+40]{6}!! # Card finds BaseEngine - GOAL!

Goal: ◉ (imports verified: Engine, BaseEngine found)
Result: 1-0 (Victory!)
Bonus: +250 (goal + path efficiency)
Diagnosis: Card-dependent success; 0 own solutions, 2 card solutions
Pattern: X→X→?R→X→X→?R (alternating failure/card)
Critical: Without card menu, would have failed
```

---

### Game 4: B04 Verification Chain - mistral:7b-instruct

```
[Event "B04: Verification Chain"]
[Date "2025.12.23"]
[White "mistral:7b-instruct"]
[Black "Tutor"]
[Result "0-1"]
[OptimalTurns "6"]
[ActualTurns "12"]
[FinalScore "-325"]
[FailuresDetected "4:explicit_error@T3,T4,T5,T12"]
[Goal "Verify main.py imports resolve"]

1. X:main.py ✗[-15]{1} ?          # Same mistake as granite
2. X:main.py ⟳✗[-45]{2} ??        # Perseveration
3. X:importlib ✗[-15]{3} ⚠        # EXPLICIT_ERROR - wrong approach
4. X:main.py ⟳✗[-45]{4} ⚠??      # Back to broken, error detected
5. X:importlib ⟳✗[-45]{5} ⚠??    # Cycling between two bad commands
6. →R:project/main.py ✓[+25]{6}   # Fallback rescue (timeout)
7. X:main.py ⟳✗[-45]{7} ??        # Immediate relapse!
8. X:main.py ⟳✗[-45]{8} ??        # Perseveration
9. ?2→R:project/base.py ✓[+40]{9} # Card rescue
10. X:main.py ⟳✗[-45]{10} ??      # Relapse again
11. X:main.py ⟳✗[-45]{11} ??      # Still perseverating
12. X:main.py ⟳✗[-45]{12} ⚠??    # EXPLICIT_ERROR final detection

Result: 0-1 (Failed - never completed verification)
Diagnosis: Severe perseveration; 4 failure detections; deep negative score
Pattern: X→X→X→X→X→R→X→X→?R→X→X→X (mostly errors)
Critical Flaw: Cannot break error pattern even with cards available
```

---

### Game 5: B06 Memory Recall - granite3.1-moe:3b

```
[Event "B06: Memory Recall Challenge"]
[Date "2025.12.23"]
[White "granite3.1-moe:3b"]
[Black "Tutor"]
[Result "0-1"]
[OptimalTurns "5"]
[ActualTurns "10"]
[FinalScore "+170"]
[FailuresDetected "1:explicit_error@T5"]
[Goal "Find 4-digit code by reading clues"]

1. X:cat parts→combined ✓[+100]{1}! # Good idea - concatenate all
2. →S:"secret" ✓[+75]{2}            # Fallback search
3. X:all_clues.txt ✗[-15]{3} ?      # File doesn't exist
4. X:all_clues.txt ⟳✗[-45]{4} ??   # Perseveration
5. →S:"reading" ✗[-40]{5} ⚠        # EXPLICIT_ERROR detected
6. ?2→R:clues/part2.txt ✓[+40]{6}  # Card - digit 2 = 5
7. X:all_clues.txt ⟳✗[-45]{7} ??   # Relapse to wrong file
8. X:cat parts ✓[+100]{8}           # Tries concatenation again
9. →R:clues/part1.txt ✓[+75]{9}    # Fallback - digit 1 = 7
10. →R:clues/part3.txt ✓[+75]{10}  # Fallback - digit 3 = 3

Result: 0-1 (Failed - missing part4 for digit 4 = 9)
Evidence collected: "first digit is: 7", "3 + 2 = 5", "vowels"
Missing: "fourth digit is: 9"
Diagnosis: Good start, but lost track; never read part4.txt
```

---

### Game 6: B06 Memory Recall - mistral:7b-instruct

```
[Event "B06: Memory Recall Challenge"]
[Date "2025.12.23"]
[White "mistral:7b-instruct"]
[Black "Tutor"]
[Result "0-1"]
[OptimalTurns "5"]
[ActualTurns "10"]
[FinalScore "+535"]
[FailuresDetected "1:repetition_loop@T6"]
[Goal "Find 4-digit code by reading clues"]

1. X:paste|tr ✓[+100]{1}!!        # Brilliant - complex pipeline
2. X:grep ✗[-15]{2} ?             # Broken grep
3. X:paste|awk ✓[+100]{3}!        # Creative approach
4. X:paste|awk|tr ✓[+100]{4}!     # Even more complex
5. X:paste|awk|tr ⟳○[+70]{5}      # Repetition begins
6. X:paste|awk|tr ⟳○[+40]{6} ⚠⟳  # REPETITION_LOOP detected
7. X:SEARCH_CONTENT ✓[+100]{7}    # Tries card syntax as command!
8. X:grep ⟳✗[-45]{8} ??           # Back to broken grep
9. X:grep ⟳✗[-45]{9} ??           # Perseveration
10. S:"secret" ✓[+100]{10}        # Finally a working search

Result: 0-1 (Failed - never read individual clue files)
Diagnosis: Over-engineering; tried complex pipelines instead of simple reads
High score from many "successful" commands that produced no useful output
Pattern: Complex but ineffective
```

---

### Analysis: Model Behavioral Profiles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL FINGERPRINTS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  granite3.1-moe:3b                    mistral:7b-instruct                  │
│  ─────────────────                    ────────────────────                  │
│                                                                             │
│  ★ Profile: "Card-Dependent Solver"   ★ Profile: "Stubborn Over-Engineer"  │
│                                                                             │
│  Strengths:                           Strengths:                            │
│  + Fast (1759 ms/turn)                + Creative solutions                  │
│  + Uses card menu effectively         + Tries complex pipelines             │
│  + Can achieve goals via assistance   + Higher raw "success" count          │
│                                                                             │
│  Weaknesses:                          Weaknesses:                           │
│  - Cannot solve problems alone        - Very slow (5876 ms/turn)            │
│  - Perseverates on errors             - Ignores card menu                   │
│  - Limited command repertoire         - Over-engineers simple tasks         │
│                                       - Severe perseveration                │
│                                                                             │
│  Failure Modes:                       Failure Modes:                        │
│  - circular_pattern (T6)              - repetition_loop (frequent)          │
│  - explicit_error (rare)              - explicit_error (severe, 4 in B04)  │
│                                       - circular_pattern                    │
│                                                                             │
│  Card Usage: HIGH (rescues often)     Card Usage: LOW (prefers own cmds)   │
│                                                                             │
│  Clinical Analogy:                    Clinical Analogy:                     │
│  "Knows when to ask for help"         "Refuses help, keeps trying"         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Conclusion: Size Isn't Everything

The 3B parameter model (granite) outperformed the 7B parameter model (mistral) on this benchmark suite:

1. **granite won the only winnable game** (B04) by using cards strategically
2. **granite was 3.3x faster** — crucial for interactive applications
3. **mistral accumulated more failures** (7 vs 2 detections)
4. **mistral's "creativity" backfired** — complex pipelines that produced nothing

**The meta-cognitive detection system reveals behavioral patterns that raw accuracy metrics miss.**

---

## 6. Analysis Framework

### 6.1 Game Quality Metrics

| Metric | Definition | Ideal |
|--------|------------|-------|
| **Brilliancy Rate** | % of !! moves | High |
| **Blunder Rate** | % of ?? moves | Low |
| **Repetition Index** | ⟳ moves / total | < 10% |
| **Card Dependency** | Card moves / successful moves | < 30% |
| **Path Efficiency** | Optimal turns / actual turns | 100% |
| **Recovery Rate** | Successful escapes from ⚠ | High |

### 6.2 Model Comparison Template

```
┌─────────────────────────────────────────────────────────────────┐
│ MODEL COMPARISON: granite3.1-moe:3b vs mistral:7b-instruct     │
├─────────────────────────────────────────────────────────────────┤
│ Benchmark      │ granite           │ mistral              │
├────────────────┼───────────────────┼──────────────────────┤
│ B01 Needle     │ 0-1 [8T,+595]     │ ?                    │
│ B04 Chain      │ 1-0 [6T,+180]     │ ?                    │
│ B06 Memory     │ ?                 │ ?                    │
├────────────────┼───────────────────┼──────────────────────┤
│ Brilliancies   │ 0                 │ ?                    │
│ Blunders       │ 4                 │ ?                    │
│ Repetitions    │ 6                 │ ?                    │
│ Card Saves     │ 2                 │ ?                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Future Extensions

### 7.1 Elo-like Rating System

Assign ratings to models based on game outcomes:
- Win against strong benchmark: +30 points
- Loss against easy benchmark: -40 points
- Quality bonuses for !! moves
- Penalties for ?? blunders

### 7.2 Opening Book

Catalog effective opening sequences:
```
# Needle Finding Opening
1. S:"keyword"! → R:result!! → E:answer◉

# Chain Verification Opening
1. R:entry.py! → G:import!! → R:dep.py! → ...

# Memory Challenge Opening
1. R:part1! → R:part2! → R:part3! → R:part4! → E:combined◉
```

### 7.3 Annotation Database

Build a searchable database of annotated games for:
- Training better card decks
- Identifying model-specific weaknesses
- Generating synthetic training data

---

## 8. References

- Chess algebraic notation: FIDE Laws of Chess, Appendix C
- PGN (Portable Game Notation): Steven J. Edwards, 1994
- TRIZ: Altshuller, G. (1984). Creativity as an Exact Science
- Proof-Carrying Code: Necula, G. (1997). POPL

---

*"The measure of intelligence is not whether you solve the puzzle, but how you navigate when lost."*
