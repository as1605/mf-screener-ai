# Documentation and Explainability

Create a documentation to explain this project, for both an investor looking to understand how the strategies are made, and for a developer who wishes to run this on their machine.

The documentation should look good both on GitHub repo view and GitHub Pages keeping / as root.
Use Jekyll `include` syntax for inserting markdowns for web.

Keep this structure
- README.md : Introduction of the project and navigation. This should be like a homepage for the project, also give link to the spreadsheet. Explain the overall workflow to a non-technical person who knows about AI/vibecoding. Include disclaimers for financial compliance. Acknowledge TickerTape for the API and Cursor for the vibecoding help.
- docs/TASKS.md : Explain briefly how the repo was made and what each task does logically.
- docs/SETUP.md : How to setup for a developer, mention use of cursor, also google sheet api steps. Also explain folder structure. Explain how to rerun 2-algorithm.md with updated prompt to get new custom algorithm for sector of choice
- docs/algorithms/{Sector}_{model}.md : Explain the strategy used by the script from `src/algorithms/{Sector}_{model}.py` along with basic results (top 5). Highlight the data points and main metrics used. Do not go too much deep into the financial justification. Also provide relative link to the result csv
- docs/algorithms/{Sector}.md : Navigation and overall strategy and overall results (top 5). Also provide relative link to the result csv. Insert markdowns in web.
- docs/ALGORITHMS.md: Navigation and general idea/introduction with a disclaimer that these are AI generated not certified financials.

If rerunning, focus on revalidating the algorithms and updating changes to the logic.

Make sure the whole documentation structure is easy to read and well formatted


Encourage developers to fork the repo, modify 2_algorithm.md and rerun