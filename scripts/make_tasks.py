#!/usr/bin/env python3
"""Generate the fixed 40-task auto-scored suite. Deterministic (seeded).
Task types: extraction (12), arithmetic (10), format (10), summarise-keyword (8).
Each task: id, type, prompt, max_tokens, checker spec.
Scoring is in score.py; this file + suite are committed BEFORE any model runs."""
import json, random

random.seed(20261023)
tasks = []

# ---- extraction: answer is a verbatim span in the prompt; exact match (normalized)
people = [("Marta Ellison", "Denver", "accountant", "2016"), ("Jonas Feld", "Austin", "plumber", "2019"),
          ("Priya Nair", "Seattle", "teacher", "2012"), ("Tom Okafor", "Chicago", "electrician", "2021"),
          ("Lena Vogt", "Portland", "nurse", "2018"), ("Omar Haddad", "Miami", "chef", "2015"),
          ("Grace Liu", "Boston", "lawyer", "2020"), ("Peter Novak", "Phoenix", "mechanic", "2011"),
          ("Sara Lindqvist", "Denver", "designer", "2017"), ("David Mensah", "Atlanta", "driver", "2013"),
          ("Anna Rossi", "Chicago", "engineer", "2022"), ("Karl Jensen", "Seattle", "farmer", "2010")]
for i, (name, city, job, year) in enumerate(people):
    other = people[(i + 3) % len(people)]
    text = (f"{name} is a {job} who lives in {city} and started the job in {year}. "
            f"{other[0]} is a {other[2]} who lives in {other[1]} and started in {other[3]}.")
    q, ans = [("In which city does " + name.split()[0] + " live?", city),
              ("What is " + name.split()[0] + "'s job?", job),
              ("In what year did " + name.split()[0] + " start the job?", year)][i % 3]
    tasks.append({"id": f"ext{i+1:02d}", "type": "extraction",
                  "prompt": f"Read the text and answer with only the answer, nothing else.\nText: {text}\nQuestion: {q}\nAnswer:",
                  "max_tokens": 12, "check": {"kind": "exact", "answer": ans}})

# ---- arithmetic: exact numeric match (last number in output must equal answer)
nums = [(17, 25), (134, 87), (46, 19), (58, 6), (231, 145), (97, 38), (12, 14), (9, 23), (64, 27), (150, 75)]
for i, (a, b) in enumerate(nums):
    if i % 2 == 0:
        tasks.append({"id": f"ari{i+1:02d}", "type": "arithmetic",
                      "prompt": f"Compute {a} + {b}. Reply with only the number.\nAnswer:",
                      "max_tokens": 10, "check": {"kind": "number", "answer": a + b}})
    else:
        tasks.append({"id": f"ari{i+1:02d}", "type": "arithmetic",
                      "prompt": f"Compute {a} * {b}. Reply with only the number.\nAnswer:",
                      "max_tokens": 10, "check": {"kind": "number", "answer": a * b}})

# ---- format-following: structural checks
fmt = [
    ("Reply with exactly the word PINEAPPLE and nothing else.", {"kind": "exact", "answer": "PINEAPPLE"}),
    ("List three colors, one per line, each line starting with '- '.", {"kind": "bullets", "n": 3}),
    ("Write a JSON object with keys \"name\" and \"age\" for a person named Rosa who is 34. Output only the JSON.",
     {"kind": "json_keys", "keys": ["name", "age"], "values": {"name": "Rosa", "age": 34}}),
    ("Say the word hello in all capital letters, five times, separated by single spaces.", {"kind": "exact", "answer": "HELLO HELLO HELLO HELLO HELLO"}),
    ("Write exactly four words that name seasons.", {"kind": "word_count", "n": 4}),
    ("Output a JSON array containing the numbers 1, 2 and 3 in order. Output only the JSON.", {"kind": "json_array", "answer": [1, 2, 3]}),
    ("Answer this question with a single word: What color is the sky on a clear day?", {"kind": "word_count_max", "n": 1}),
    ("Write the numbers from 1 to 5 separated by commas, like 1,2,3. Nothing else.", {"kind": "contains", "answer": "1,2,3,4,5"}),
    ("List two fruits, one per line, each line starting with '- '.", {"kind": "bullets", "n": 2}),
    ("Write a JSON object with the single key \"capital\" whose value is \"Paris\". Output only the JSON.",
     {"kind": "json_keys", "keys": ["capital"], "values": {"capital": "Paris"}}),
]
for i, (p, chk) in enumerate(fmt):
    tasks.append({"id": f"fmt{i+1:02d}", "type": "format", "prompt": p + "\nResponse:",
                  "max_tokens": 48, "check": chk})

# ---- summarise-keyword: output must contain all keywords (case-insensitive), <= 60 words
summ = [
    ("The city council voted on Tuesday to build a new library downtown. The project will cost 12 million dollars "
     "and create 200 jobs. Construction begins in March and the library is expected to open in 2027.",
     ["library", "2027"]),
    ("Scientists reported that the new vaccine reduced infections by 60 percent in trials with 4000 volunteers. "
     "The vaccine was developed by a team in Norway and targets a virus that spreads in winter.",
     ["vaccine", "norway"]),
    ("The football team won the championship after scoring two goals in the final ten minutes. The captain, "
     "Maria Santos, was named player of the match. Fans celebrated in the stadium in Lisbon.",
     ["championship", "lisbon"]),
    ("A local bakery in Vermont began selling sourdough bread made with apples from nearby orchards. The owner "
     "says the bread sells out every morning and she plans to hire two more bakers.",
     ["bakery", "sourdough"]),
    ("The train line between Oslo and Bergen was closed on Friday after heavy snow blocked the tracks. Buses "
     "replaced trains for the weekend and engineers cleared the line by Monday.",
     ["train", "snow"]),
    ("Researchers at a university in Japan built a robot that can fold laundry. The robot uses cameras and "
     "small grippers and can fold a shirt in under a minute.",
     ["robot", "laundry"]),
    ("The farmer's market in Austin will move to a larger site next month because of crowding. The new site "
     "has space for 80 stalls and a parking area for visitors.",
     ["market", "austin"]),
    ("A storm damaged the lighthouse on the coast of Maine on Sunday night. The light was repaired on "
     "Wednesday and no ships were harmed during the outage.",
     ["lighthouse", "storm"]),
]
for i, (text, kws) in enumerate(summ):
    tasks.append({"id": f"sum{i+1:02d}", "type": "summarise",
                  "prompt": f"Summarise the following text in at most 30 words.\nText: {text}\nSummary:",
                  "max_tokens": 64, "check": {"kind": "keywords", "keywords": kws, "max_words": 60}})

assert len(tasks) == 40, len(tasks)
with open("tasks.json", "w") as f:
    json.dump(tasks, f, indent=1)
print("wrote tasks.json:", len(tasks), "tasks")
