# Project Requirements (from `report_guide.tex`)

## Submission and format
- Minimum 8 pages of content in ACL format (template provided), excluding AI disclosure and references; no filler.
- Deliverables: PDF to Gradescope + email with code link (public repo or zip <10MB). Include URL in report.
- Use ACL style; Overleaf/LaTeX expected. Include multiple tables/plots conveying numerical info.

## Required sections (must cover)
- Problem statement: goal/motivation summary.
- Proposed vs. accomplished: bulleted list of proposal items with status and brief reasons for misses.
- Related work: cohesive narrative citing ≥6 papers (not a list of summaries); cite properly.
- Dataset: source, stats, examples, I/O pairs, why task is challenging.
  - Preprocessing; annotation + IAA if applicable.
- Baselines: what they are, how they work, hyperparams, tuning, splits (train/val/test).
- Approach: model/strategy details, what you implemented, file paths, libraries, hardware/runtime notes, Colab hacks, comparison to baselines.
- Results: quantitative results with statistical significance (CI/t-tests/bootstrap/paired comparisons as appropriate).
- Error analysis: manual analysis of failures (e.g., annotate ~100 failed examples, discuss patterns).
- Contributions of group members: per-person roles.
- Conclusion: takeaways, surprises, future directions.
- AI disclosure: list AIs used, all prompts, and experience using them.

## Gaps vs. current plan/implementation
- Significance testing not planned for Phase 3 results; need CI/paired tests on shift metrics.
- No manual error analysis/human review of failures (strategy adherence, emotion mislabels).
- Baseline definitions not explicit (e.g., simple empathetic baseline vs. strategy prompts; neutral/no-strategy control).
- Dataset/examples section needs concrete stats and input/output snippets (ESConv seeds, simulations).
- Contributions/AI disclosure not yet drafted.
- Multi-turn/policy phases (4/5) still pending; if omitted, must be clearly marked as uncompleted in “proposed vs accomplished.”

## Actions to comply
- Run Phase 3 with multiple seeds/runs and report CIs (binomial for proportions, bootstrap/paired tests for comparisons).
- Add a simple baseline (e.g., generic supportive reply without strategy) to compare against strategy-conditioned responses.
- Perform manual error analysis on a sample of failures (e.g., strategy outputs that mis-shift emotion or violate prompt).
- Collect dataset stats/examples and describe preprocessing; include representative conversation snippets and classified outputs.
- Draft related work with ≥6 citations tied to our narrative (emotion flow, empathetic response, persuasion/strategy).
- Fill contributions and AI disclosure sections; log any AI usage and prompts.
- If multi-turn/policy not done, document as missing with reasons; optionally add a small multi-turn demo to reduce the gap.
