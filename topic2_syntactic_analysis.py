"""
Topic 2: Syntactic Analysis of Social Media Text
Tasks: (1) Part-of-Speech (POS) tagging  (2) Dependency parsing

Setup:
    pip install spacy pandas matplotlib
    python -m spacy download en_core_web_sm

Run:
    python topic2_syntactic_analysis.py

Outputs:
    topic2_pos_tags.csv
    topic2_dependencies.csv
    topic2_parse_post1.svg
    topic2_parse_post3.svg
    topic2_parse_post6.svg

Note:
    The posts below are a manually prepared sample dataset for this
    assignment. They are not presented as scraped or representative
    social-media data.
"""

import re
from pathlib import Path

import pandas as pd
import spacy
from spacy import displacy


# ---------------------------------------------------------------
# 1. Sample dataset: social media posts (mix of clean and noisy)
# ---------------------------------------------------------------
posts = [
    "The new phone camera is absolutely amazing!",
    "Just landed in Nairobi and the weather is lovely.",
    "I can't believe the airline lost my luggage again.",
    "Customer service solved my problem quickly. Great job!",
    "This update ruined the app and nobody fixed it.",
    "omg the concert last night was sooo good lol",
    "Traffic on Mombasa Road is terrible today #nightmare",
    "@SafaricomPLC why is my internet so slow???",
    "Loving the new battery life, best upgrade ever :)",
    "The delivery arrived late and the box was damaged.",
]


# ---------------------------------------------------------------
# 2. Light cleaning
#    Remove URLs only; retain hashtags, mentions and informal text.
#    spaCy may tokenize a hashtag marker (#) separately from its word.
# ---------------------------------------------------------------
def clean(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    return text.strip()


# Load the English pipeline once and process all posts.
nlp = spacy.load("en_core_web_sm")
docs = [nlp(clean(post)) for post in posts]


# ---------------------------------------------------------------
# 3. Part-of-Speech (POS) tagging
#    token.pos_ = coarse Universal POS tag
#    token.tag_ = fine-grained Penn Treebank tag
# ---------------------------------------------------------------
print("=" * 70)
print("PART 1: POS TAGGING")
print("=" * 70)

pos_rows = []

for i, doc in enumerate(docs, 1):
    print(f"\nPost {i}: {doc.text}")
    print(f"{'TOKEN':<14}{'UPOS':<8}{'PENN TAG':<10}{'EXPLANATION'}")

    for tok in doc:
        explanation = spacy.explain(tok.tag_) or "No explanation available"
        print(f"{tok.text:<14}{tok.pos_:<8}{tok.tag_:<10}{explanation}")

        pos_rows.append(
            {
                "post": i,
                "token": tok.text,
                "upos": tok.pos_,
                "tag": tok.tag_,
                "explanation": explanation,
            }
        )

pos_df = pd.DataFrame(pos_rows)
pos_df.to_csv("topic2_pos_tags.csv", index=False)

print("\nPOS distribution across all posts:")
print(pos_df["upos"].value_counts().to_string())


# ---------------------------------------------------------------
# 4. Dependency parsing
#    token.dep_ = syntactic relation to its head
#    token.head = governing word
#    token.children = words directly attached to the token
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: DEPENDENCY PARSING")
print("=" * 70)

dep_rows = []

for i, doc in enumerate(docs, 1):
    print(f"\nPost {i}: {doc.text}")
    print(f"{'TOKEN':<14}{'DEP':<12}{'HEAD':<14}{'CHILDREN'}")

    for tok in doc:
        children = [child.text for child in tok.children]
        children_text = ", ".join(children)

        print(f"{tok.text:<14}{tok.dep_:<12}{tok.head.text:<14}{children}")

        dep_rows.append(
            {
                "post": i,
                "token": tok.text,
                "dep": tok.dep_,
                "head": tok.head.text,
                "children": children_text,
            }
        )

dep_df = pd.DataFrame(dep_rows)
dep_df.to_csv("topic2_dependencies.csv", index=False)

print("\nMost common dependency relations:")
print(dep_df["dep"].value_counts().head(10).to_string())


# ---------------------------------------------------------------
# 5. Subject-verb-object and subject-verb-complement patterns
#    spaCy versions may use "dobj" or "obj" for direct objects.
#    "attr" and "acomp" identify complements, not direct objects.
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("SUBJECT-VERB-OBJECT / COMPLEMENT PATTERNS")
print("=" * 70)

subject_relations = ("nsubj", "nsubjpass")
object_complement_relations = ("dobj", "obj", "attr", "acomp")

for i, doc in enumerate(docs, 1):
    for tok in doc:
        if tok.pos_ in ("VERB", "AUX") and tok.dep_ in ("ROOT", "conj", "ccomp"):
            subjects = [
                child.text
                for child in tok.children
                if child.dep_ in subject_relations
            ]
            objects_complements = [
                child.text
                for child in tok.children
                if child.dep_ in object_complement_relations
            ]

            if subjects or objects_complements:
                print(
                    f"Post {i}: {subjects} -> {tok.lemma_} "
                    f"-> {objects_complements}"
                )


# ---------------------------------------------------------------
# 6. Dependency diagrams (SVG files for the report)
# ---------------------------------------------------------------
for idx in (0, 2, 5):  # posts 1, 3 and 6
    svg = displacy.render(
        docs[idx],
        style="dep",
        jupyter=False,
        options={"compact": False, "distance": 110},
    )
    Path(f"topic2_parse_post{idx + 1}.svg").write_text(
        svg,
        encoding="utf-8",
    )


# ---------------------------------------------------------------
# 7. Noise analysis: inspect tagging of informal social-media text
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("NOISY TEXT CHECK (POSTS 6, 7 AND 8)")
print("=" * 70)

for idx in (5, 6, 7):
    print(f"Post {idx + 1}: {[(token.text, token.pos_) for token in docs[idx]]}")


print(
    "\nSaved: topic2_pos_tags.csv, topic2_dependencies.csv, "
    "topic2_parse_post1.svg, topic2_parse_post3.svg, "
    "topic2_parse_post6.svg"
)
