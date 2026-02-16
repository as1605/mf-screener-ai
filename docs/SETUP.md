# Setup Guide

## 🚀 Fork and Create Your Own Algorithm

**This project is designed to be forked and customized!** The core value proposition is enabling developers to create their own AI-generated mutual fund scoring algorithms.

### Why Fork?

1. **Different Perspectives**: Each AI model (Claude, Gemini, Codex) brings unique insights. What will your model discover?
2. **Custom Sectors**: Want to analyze Large Cap funds? Sector-specific funds? Just modify `tasks/2_algorithm.md` and generate a new algorithm.
3. **Learning Experience**: Understand how quantitative fund analysis works by seeing how AI models approach the problem.
4. **Open Source Contribution**: Share your findings and algorithms with the community.

### Quick Start: Create Your Own Algorithm

1. **Fork this repository** on GitHub
2. **Clone your fork** locally
3. **Follow the "Customizing Algorithms" section below** to modify `tasks/2_algorithm.md`
4. **Use Cursor** (or your preferred AI coding assistant) to generate a new scoring script
5. **Run and compare** your results with existing models

The beauty of this approach is that **you don't need deep financial expertise** - the AI models do the research and implementation. You just need to guide them with the right prompt!

---

## Prerequisites

*   Python 3.10+
*   Cursor (Recommended for vibecoding)
*   Google Cloud Service Account (for Sheets API)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd mf-screener-ai
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Environment Variables**:
    Copy `sample.env` to `.env` and fill in the details:
    ```bash
    cp sample.env .env
    ```
    *   `GOOGLE_SERVICE_ACCOUNT_KEY`: Path to your Google Service Account JSON key file.
    *   `GOOGLE_SHEET_URL`: URL of the Google Sheet where results will be published.

2.  **Google Sheets API**:
    *   Enable Google Sheets API and Google Drive API in your Google Cloud Console.
    *   Create a Service Account and download the JSON key.
    *   Share your target Google Sheet with the Service Account email address.

## Folder Structure

*   `src/`: Source code for data provider, algorithms, and publishing scripts.
    *   `algorithms/`: Contains the AI-generated scoring scripts.
*   `data/`: Stores fetched data (cached by date).
*   `results/`: Stores the output CSV files from algorithms.
*   `tasks/`: Task descriptions used to build the project.
*   `docs/`: Documentation files.

## Running Algorithms

To generate a new custom algorithm or rerun existing ones:

1.  **Update Data**:
    Ensure you have the latest data. The `MfDataProvider` will fetch it if not present for the current date.

2.  **Run an Algorithm**:
    ```bash
    python src/algorithms/Small\ Cap_Claude.py
    ```
    This will generate `results/Small Cap_Claude.csv`.

3.  **Compile and Publish**:
    ```bash
    python src/publish_sheet.py
    ```
    This will compile all results in `results/` and upload them to the Google Sheet.

## Customizing Algorithms (Vibecoding)

**This is the core workflow of the project!** Creating your own algorithm is straightforward:

### Step-by-Step Guide

1.  **Fork and Clone**:
    ```bash
    # Fork the repo on GitHub first, then:
    git clone https://github.com/YOUR_USERNAME/mf-screener-ai.git
    cd mf-screener-ai
    ```

2.  **Open the Task File**:
    Open `tasks/2_algorithm.md` in Cursor.

3.  **Update the Prompt Variables**:
    Modify the variables at the top of the file:
    ```markdown
    {SECTOR}={Your Sector Here}        # e.g., "Large Cap", "Technology", "Debt"
    {model}={Your Model Here}          # e.g., "GPT4", "Claude", "Gemini", "YourName"
    ```

4.  **Generate Code with Cursor**:
    - Open Cursor's Chat or Composer
    - Reference the task file: `@tasks/2_algorithm.md`
    - Ask: "Implement this task for the {SECTOR} sector using the {model} model."
    - The AI will research financial metrics, design the strategy, and generate the complete Python script

5.  **Review the Generated Code**:
    - The script will be created in `src/algorithms/{SECTOR}_{model}.py`
    - Review the logic, metrics used, and scoring approach
    - The AI does the heavy lifting - you just need to verify it makes sense

6.  **Run Your Algorithm**:
    ```bash
    python src/algorithms/Your\ Sector_YourModel.py
    ```
    This generates `results/Your Sector_YourModel.csv`

7.  **Compare Results**:
    Compare your algorithm's top picks with existing models. Do they agree? What's different?

8.  **Share Your Findings**:
    - Create a PR to share your algorithm
    - Document your strategy in `docs/algorithms/`
    - Discuss your approach in issues or discussions

### Example: Creating a Large Cap Algorithm

```bash
# 1. Edit tasks/2_algorithm.md:
{SECTOR}={Large Cap}
{model}={GPT4}

# 2. In Cursor Chat:
@tasks/2_algorithm.md
"Implement this task for Large Cap sector using GPT4 model"

# 3. Run the generated script:
python src/algorithms/Large\ Cap_GPT4.py

# 4. Check results:
cat results/Large\ Cap_GPT4.csv
```

### Tips for Success

- **Be Specific**: The more context you provide in the prompt, the better the algorithm
- **Review Existing Algorithms**: Look at `src/algorithms/` to understand different approaches
- **Iterate**: Don't be afraid to refine the prompt and regenerate
- **Compare**: Always compare your results with existing models to validate your approach
- **Document**: Create documentation for your algorithm following the pattern in `docs/algorithms/`

### What Makes This Special?

Unlike traditional quantitative finance projects that require deep domain expertise, this approach lets you:
- **Leverage AI Research**: The models research financial metrics and strategies for you
- **Focus on Prompting**: Your job is to guide the AI with good prompts, not write complex financial code
- **Rapid Experimentation**: Generate multiple algorithms quickly and compare them
- **Learn by Example**: See how different AI models approach the same problem differently
