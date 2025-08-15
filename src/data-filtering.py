import argparse
import asyncio
import json
import os
import re
from tqdm import tqdm
from diskcache import Cache

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

CACHE = Cache("./llm_cache")
CHECKPOINT_EVERY = 10

# ---------------------------------------------------------------------------
# Prompt contexts
# ---------------------------------------------------------------------------

FILTER_CONTEXT = """
You are an educational assistant. Your task is to determine whether a math problem
is appropriate for a student in elementary school grades 3 to 5 (ages 5 to 9). These students should be
working with:

Simple math concepts such as:

- Addition, subtraction, multiplication, division
- Whole numbers, fractions, decimals, percentages
- Familiar Geometric shapes, basic area and perimeter
- Simple child-level Data in tables or simple graphs
- Simple child-level Measurement (time, volume, length)
- Simple child-level reasoning with simple equalities and inequalities
- Simple child appropiate pattern recognition
- Word problems with real-life context

IMPORTANT: Exclude problems that require:
- Matrix handling
- sin, cos, tan, or other trigonometric functions
- Functions f(x) or g(x) etc..
- Algebraic equations or factorization
- Algebra, calculus, or advanced math concepts
- Complex number systems or modular arithmetic
- Advanced proofs, exponential or logarithmic functions
- Functions or graphing beyond basic linear relationships
- Numerical analysis
- Calculus, limits, derivatives, integrals
- Modular arithmetic, base systems beyond base 10
- Abstract algebra, advanced proofs, irrational numbers
- Logarithms, complex numbers, advanced symbolic manipulation

Respond with:
{{"include": true}} or {{"include": false}}
"""

SUBJECT_CONTEXT = """
You are a math curriculum assistant. Given a math problem and its solution, choose
the main **subject** from this list:

- Addition, subtraction, multiplication, division
- Whole numbers, fractions, decimals, percentages
- Geometric shapes, basic area and perimeter
- Data in tables or simple graphs
- Measurement (time, volume, length)
- Reasoning with simple equalities and inequalities
- Pattern recognition
- Word problems with real-life context

Respond ONLY with:
{{"subject": "<chosen subject>"}}
"""

OBJECTIVE_CONTEXT = """
You are a math curriculum expert. Based on the following list of pedagogical objectives,
assign the one that best matches the given math problem and its solution:

- OP01: Recognize and describe patterns and regularities in different contexts.
- OP02: Describe and represent variation using diagrams, words, or tables.
- OP03: Construct numerical equalities and inequalities from relationships.
- OP04: Solve additive or multiplicative problems using strategies or estimation.
- OP05: Solve problems involving fractions, decimals, or percentages.
- OP06: Interpret and represent data with charts or tables.
- OP07: Apply geometry to describe or analyze figures and objects in space.
- OP08: Use standard measurement units (length, area, volume, time).
- OP09: Develop logical and critical thinking in contextual math problems.
- OP10: Promote math communication through explanation and justification.

Respond ONLY with a valid JSON like:
{{"objective_code": "OP04"}}
"""

# ---------------------------------------------------------------------------
# Few-shot examples for inclusion
# ---------------------------------------------------------------------------

FEW_SHOT = [
    {
        "problem": "Ava has 5 boxes of crayons. Each box has 8 crayons. How many crayons does she have in total?",
        "generated_solution": "5 × 8 = 40. Ava has 40 crayons.",
        "include": True
    },
    {
        "problem": "If f(x) = x^2 + 3x, what is f(2)?",
        "generated_solution": "f(2) = 2^2 + 3×2 = 4 + 6 = 10.",
        "include": False
    },
    {
        "problem": "Calculate the value of √49.",
        "generated_solution": "The square root of 49 is 7.",
        "include": False
    },
    {
        "problem": "Luca drank 2/3 of his juice. What fraction of the juice is left?",
        "generated_solution": "1 - 2/3 = 1/3. One third is left.",
        "include": True
    }
]

# ---------------------------------------------------------------------------
# Initialize chains
# ---------------------------------------------------------------------------
def is_obviously_advanced(text: str) -> bool:
    return bool(re.search(r'\b(f\([a-zA-Z]\)|√|\^|log|ln|sin|cos|tan|mod|x\s*=|[a-z]\^)', text))

def _init_chains(model: str):
    llm = OllamaLLM(model=model, temperature=0)

    inc_msgs = [("system", FILTER_CONTEXT)]
    for ex in FEW_SHOT:
        inc_msgs.append(("user", f"Problem:\n{ex['problem']}\n\nSolution:\n{ex['generated_solution']}"))
        inc_msgs.append(("assistant", f'{{{{"include": {"true" if ex["include"] else "false"}}}}}'))
    inc_msgs.append(("user", "Problem:\n{problem}\n\nSolution:\n{generated_solution}"))
    include_chain = ChatPromptTemplate.from_messages(inc_msgs) | llm

    subject_chain = ChatPromptTemplate.from_messages([
        ("system", SUBJECT_CONTEXT),
        ("user", "Problem:\n{problem}\n\nSolution:\n{generated_solution}")
    ]) | llm

    objective_chain = ChatPromptTemplate.from_messages([
        ("system", OBJECTIVE_CONTEXT),
        ("user", "Problem:\n{problem}\n\nSolution:\n{generated_solution}")
    ]) | llm

    return include_chain, subject_chain, objective_chain

# ---------------------------------------------------------------------------
# Checkpoint utils
# ---------------------------------------------------------------------------

def read_checkpoint(path):
    return int(open(path).read().strip()) if os.path.exists(path) else 0

def write_checkpoint(path, i):
    with open(path, "w") as f:
        f.write(str(i))

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def evaluate(obj, chains):
    include_chain, subject_chain, objective_chain = chains
    key = json.dumps(obj, sort_keys=True)
    if key in CACHE:
        return CACHE[key]

    problem = obj.get("problem", "")
    solution = obj.get("generated_solution", "")

    if is_obviously_advanced(problem) or is_obviously_advanced(solution):
        return None


    try:
        raw = include_chain.invoke({"problem": problem, "generated_solution": solution}).strip()
        if not json.loads(re.search(r"\{.*?\}", raw).group(0)).get("include", False):
            return None
    except:
        return None

    # Classification (subject and objective)
    async def classify(chain, tag):
        try:
            return await asyncio.to_thread(chain.invoke, {
                "problem": problem,
                "generated_solution": solution
            })
        except Exception as e:
            print(f"⚠️ {tag} failed: {e}")
            return None

    subject_raw, objective_raw = await asyncio.gather(
        classify(subject_chain, "subject"),
        classify(objective_chain, "objective")
    )

    try:
        obj["subject"] = json.loads(re.search(r"\{.*?\}", subject_raw).group(0))["subject"]
    except: pass

    try:
        obj["objective_code"] = json.loads(re.search(r"\{.*?\}", objective_raw).group(0))["objective_code"]
    except: pass

    CACHE[key] = obj
    return obj

# ---------------------------------------------------------------------------
# Process file
# ---------------------------------------------------------------------------

def process(input_path, output_path, model_name):
    ckpt_path = output_path + ".ckpt"
    start_idx = read_checkpoint(ckpt_path)

    chains = _init_chains(model_name)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    async def run_all():
        with open(output_path, "a", encoding="utf-8") as fout:
            for i in tqdm(range(start_idx, len(lines)), desc="Processing"):
                try:
                    obj = json.loads(lines[i])
                except:
                    continue

                result = await evaluate(obj, chains)
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

                if (i + 1) % CHECKPOINT_EVERY == 0:
                    write_checkpoint(ckpt_path, i + 1)

            write_checkpoint(ckpt_path, len(lines))

    asyncio.run(run_all())
    print(f"\n✅ Done → {output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .jsonl")
    parser.add_argument("output", help="Output .jsonl")
    parser.add_argument("--model", default="gemma3:1b")
    args = parser.parse_args()
    process(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
