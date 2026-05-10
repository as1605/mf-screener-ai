# Tasks Overview

This repository was built by following a series of tasks, each designed to build a specific component of the system.

## 1. Data Provider (`tasks/1_data-provider.md`)
**Goal**: Create a robust data fetching utility.
-   Developed `MfDataProvider` class to fetch mutual fund metadata and historical NAV data.
-   Handles caching of data in `data/yyyy-mm-dd/` directory.
-   Provides methods to list funds, get charts, and fetch index data.

## 2. Algorithm Generation (`tasks/2_algorithm.md`)
**Goal**: Generate scoring algorithms using different AI models.
-   Prompted AI models (Claude, Gemini, GPT, Grok) to create Python scripts per sector (e.g. Small Cap, Mid Cap, Total Market, Multi Asset Allocation).
-   Each model implemented its own strategy using metrics appropriate to the mandate (single-equity-benchmark alpha vs multi-asset sleeve inference, etc.).
-   Scripts output CSV files with scores and ranks under `results/`.

## 3. Publish Results (`tasks/3_publish.md`)
**Goal**: Compile and publish the results.
-   Created scripts to compile results from multiple models into a single CSV.
-   Implemented normalization of scores to create a final composite rank.
-   Automated the publishing of results to a Google Sheet with formatting.

## 4. Documentation (`tasks/4_document.md`)
**Goal**: Document the project.
-   Created this documentation structure to explain the project to developers and investors.
