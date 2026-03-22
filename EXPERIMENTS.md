# Experiment Log

## Dataset Expansion: Justia Clauses

**Hypothesis:** Adding 868 contract clauses from Justia (13 categories) with category descriptions as queries would improve MLEB Contractual Clause Retrieval performance.

**Result:** NDCG@10 dropped from 0.7796 to 0.7335 (-5.9%).

**Why it hurt:** The Justia queries are generic category descriptions ("This clause requires one party to compensate the other for losses, damages, or liabilities.") while the MLEB benchmark uses specific semantic descriptions of clause function. The model learned to match broad categories rather than fine-grained clause semantics.

**What the Justia data IS useful for:**
- Cross-category hard negatives (confusing clause pairs from different categories)
- Evaluation corpus for cross-contract comparison
- Clause-to-clause training pairs (same category = positive, different category = negative)
- End-to-end retrieval testing with the 30 full contracts

**Next steps to try:**
1. Use Justia clauses only as negative mining corpus (don't use category names as queries)
2. Create clause-to-clause pairs: for each clause, find similar clauses in the same collection as positives
3. Use the full contracts to create more CUAD-style QA pairs with an LLM
