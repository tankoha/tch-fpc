# Topics and length (spec-v3)

## Topics

The harness gives one topic per job, in rotation, so that each dialect
gets every topic about equally often. A topic is a theme, not a fixed
task: vary the concrete task within it.

| Topic id | Theme |
|---|---|
| `algorithms` | searching, sorting, recursion, dynamic programming, greedy choices |
| `containers` | lists, stacks, queues, maps, sets; their own implementation or the library's |
| `strings` | parsing words, formatting, case and padding, simple text statistics |
| `numbers` | integer arithmetic, primes, divisors, number bases, big sums with overflow care |
| `floating` | averages, rounding, simple numerical methods, fixed-format output |
| `dates` | calendar arithmetic on fixed dates (leap years, weekdays, durations); never today's date |
| `parsing` | tokenising small expressions, CSV-like lines, key=value text given in the source |
| `objects` | classes, inheritance, polymorphism, interfaces, properties |
| `generics` | generic types and routines, constraints where the dialect has them |
| `errors` | exceptions: raising, catching, custom exception types, cleanup |
| `records` | records, enumerations, sets, case statements, variant records where allowed |
| `functional` | procedure types, method references, anonymous methods or lambdas where the dialect has them |
| `simulation` | small deterministic simulations: queues, cellular automata, bank accounts, inventories |
| `games` | rules of small games played by fixed moves: tic-tac-toe, Nim, dice with a fixed sequence |
| `geometry` | points, distances, areas, bounding boxes, intersections on integer grids |
| `matrices` | small matrices: addition, multiplication, transpose, determinant |
| `state` | state machines, simple interpreters, undo stacks |
| `text-report` | building a formatted table or report from data in the source |

## Length

Measured in tokens of the answer (explanation plus source):
- between **150 and 1500**;
- median about **600**: most samples are 30 to 90 lines of source;
- `unit` and `extend` samples tend to the upper half, `fix` samples to
  the lower half.
