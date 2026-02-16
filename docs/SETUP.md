# Setup Guide

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

To create a new algorithm using AI (Vibecoding):

1.  **Open the Task File**:
    Open `tasks/2_algorithm.md`.

2.  **Update the Prompt**:
    Modify the variables at the top of the file:
    *   `{SECTOR}`: e.g., "Mid Cap"
    *   `{model}`: e.g., "GPT4"
    *   You can also adjust the specific instructions or focus areas in the prompt description.

3.  **Generate Code**:
    Use Cursor's "Composer" or "Chat" feature.
    *   Reference the task file: `@tasks/2_algorithm.md`
    *   Ask the AI: "Implement this task for the Mid Cap sector using the GPT4 model."

4.  **Review and Save**:
    The AI will generate a new script in `src/algorithms/`. Review the logic and save the file.

5.  **Run and Publish**:
    Follow the "Running Algorithms" steps above to generate results and publish them.
